"""Bridge Discord option signals to the project's options strategy validation."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from copy import deepcopy
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict

from .config import config
from .multi_leg_validation import validate_multi_leg_strategy
from .options_parser import ParsedOptionSignal
from .polygon_options_data import validate_exact_strike_with_polygon
from .options_symbol import resolve_underlying_for_prediction
from .prediction_bridge import build_default_prediction_input

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(1, str(PROJECT_ROOT / "tools"))
CACHE_PATH = Path(__file__).resolve().parent / "options_validation_cache.json"
_CACHE_LOCK = threading.RLock()
_TASTYTRADE_EXECUTOR = ThreadPoolExecutor(
    max_workers=max(4, min(16, config.signal_worker_concurrency * 2)),
    thread_name_prefix="option-validation",
)

try:
    from src.services.tastytrade_backtester_service import run_options_backtest
except Exception as exc:  # pragma: no cover - optional service
    run_options_backtest = None
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
else:
    _IMPORT_ERROR = ""


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _dte_from_option(option: ParsedOptionSignal, origin_date: str) -> int:
    try:
        expiry = datetime.strptime(str(option.expiry_date), "%Y-%m-%d").date()
        origin = datetime.strptime(str(origin_date), "%Y-%m-%d").date()
        return max(1, (expiry - origin).days)
    except Exception:
        return max(1, config.default_option_strategy_dte)


def _dte_from_expiry(expiry_date: Any, origin_date: str, fallback: int) -> int:
    try:
        expiry = datetime.strptime(str(expiry_date), "%Y-%m-%d").date()
        origin = datetime.strptime(str(origin_date), "%Y-%m-%d").date()
        return max(1, (expiry - origin).days)
    except Exception:
        return max(1, fallback)


def build_options_strategy_input(option: ParsedOptionSignal) -> Dict[str, Any]:
    """Build the same conceptual fields used by Options Strategy Validation."""
    underlying = resolve_underlying_for_prediction(option.root)
    spi = build_default_prediction_input(underlying, config.default_option_strategy_horizon_days)
    backtest_end = date.today()
    backtest_start = backtest_end - timedelta(days=max(30, config.option_backtest_lookback_days))
    dte = _dte_from_option(option, spi["prediction_origin_date"])
    quantity = max(1, _as_int(option.quantity, int(config.default_option_qty)))
    delta_target = max(1, _as_int(option.delta_target, config.default_option_strategy_delta))
    parsed_legs = list(option.legs or [])
    if parsed_legs:
        custom_legs = []
        for parsed_leg in parsed_legs:
            direction = "short" if parsed_leg.order_action in {"open_short", "close_long"} else "long"
            leg_dte = _dte_from_expiry(parsed_leg.expiry_date, spi["prediction_origin_date"], dte)
            custom_legs.append(
                {
                    "type": "equity-option",
                    "direction": direction,
                    "quantity": quantity * max(1, int(parsed_leg.ratio_qty)),
                    "side": "call" if parsed_leg.side == "CALL" else "put",
                    "daysUntilExpiration": leg_dte,
                    "strikeSelection": "strike",
                    "strikePrice": float(parsed_leg.strike),
                }
            )
        direction = custom_legs[0]["direction"]
        side = custom_legs[0]["side"]
        opt_type_label = "Multi-leg"
        direction_label = str(option.structure or "Multi-leg").replace("_", " ").title()
        strike_selection = "strike"
        requested_strike: Any = [leg.strike for leg in parsed_legs]
        strike_price: Any = requested_strike
    else:
        direction_label = "Buy"
        direction = "long"
        opt_type_label = "Call" if str(option.side).upper() == "CALL" else "Put"
        side = opt_type_label.lower()
        strike_selection = "strike" if option.strike is not None else "delta"
        leg = {
            "type": "equity-option",
            "direction": direction,
            "quantity": quantity,
            "side": side,
            "daysUntilExpiration": dte,
            "strikeSelection": strike_selection,
        }
        if option.strike is not None:
            leg["strikePrice"] = float(option.strike)
        else:
            leg["delta"] = delta_target
        custom_legs = [leg]
        requested_strike = option.strike
        strike_price = option.strike

    return {
        "symbol": underlying,
        "original_root": option.root,
        "historical_context_start_date": spi["historical_context_start_date"],
        "prediction_origin_date": spi["prediction_origin_date"],
        "target_date": spi["target_date"],
        "decision_horizon_days": spi["decision_horizon_days"],
        "initial_capital": spi["initial_capital"],
        "benchmark": spi["benchmark"],
        "validation_mode": spi["validation_mode"],
        "price_basis": spi["price_basis"],
        "options_backtest_start_date": backtest_start.strftime("%Y-%m-%d"),
        "options_backtest_end_date": backtest_end.strftime("%Y-%m-%d"),
        "direction_label": direction_label,
        "opt_type": opt_type_label,
        "direction": direction,
        "side": side,
        "dte": dte,
        "delta": delta_target,
        "legs": len(custom_legs),
        "entry_frequency": config.default_option_entry_frequency,
        "exit_rule": config.default_option_exit_rule,
        "strike_selection": strike_selection,
        "requested_strike": requested_strike,
        "strike_price": strike_price,
        "expiry_date": option.expiry_date,
        "leg_expiry_dates": [leg.expiry_date for leg in parsed_legs],
        "quantity": quantity,
        "price_effect": option.price_effect,
        "structure": option.structure,
        "custom_legs": custom_legs,
    }


def _decision_from_backtest(result: Dict[str, Any]) -> str:
    if result.get("status") != "SUCCESS":
        return "REVIEW"
    profit_loss = _as_float(
        result.get("profit_loss", result.get("total_profit_loss", result.get("pnl")))
    )
    if profit_loss > 0:
        return "BUY"
    if profit_loss < 0:
        return "SELL"
    return "HOLD"


def _is_empty_backtest_result(result: Dict[str, Any]) -> bool:
    message = str(result.get("message") or result.get("error") or "").lower()
    return (
        result.get("status") == "VALIDATION_FAILED"
        and (
            "statistics=null" in message
            or "trials=null/empty" in message
            or "no trial data" in message
            or "no statistics" in message
            or "all profitloss values are 0" in message
            or "misconfigured payload" in message
        )
    )


def _is_rate_limited_result(result: Dict[str, Any]) -> bool:
    message = str(result.get("message") or result.get("error") or "").lower()
    return "429" in message or "rate limit" in message or "too many requests" in message


def _read_cache_unlocked() -> Dict[str, Any]:
    try:
        if CACHE_PATH.exists():
            loaded = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                return loaded
            raise ValueError("cache root must be a JSON object")
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Ignoring unreadable options-validation cache %s: %s",
            CACHE_PATH,
            exc,
        )
    return {}


def _read_cache() -> Dict[str, Any]:
    with _CACHE_LOCK:
        return _read_cache_unlocked()


def _write_cache(cache: Dict[str, Any]) -> None:
    payload = json.dumps(cache, indent=2, default=str)
    with _CACHE_LOCK:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = CACHE_PATH.with_name(
            f"{CACHE_PATH.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, CACHE_PATH)
        finally:
            temporary.unlink(missing_ok=True)


def _prune_expired_entries(cache: Dict[str, Any]) -> Dict[str, Any]:
    ttl_seconds = max(1, config.option_validation_cache_ttl_hours) * 3600
    now = time.time()
    return {
        key: entry
        for key, entry in cache.items()
        if now - float((entry or {}).get("created_at", 0)) <= ttl_seconds
    }


def _store_cache_entry(key: str, result: Dict[str, Any]) -> None:
    """Merge one result atomically so concurrent validations cannot lose entries.

    Expired entries are dropped on every write -- the cache key embeds a
    rolling backtest date window, so without pruning it grows without bound
    (one stale entry per symbol/strategy/day forever) instead of staying
    bounded to roughly what's actually still within the TTL.
    """
    with _CACHE_LOCK:
        cache = _prune_expired_entries(_read_cache_unlocked())
        cache[key] = {"created_at": time.time(), "result": result}
        _write_cache(cache)


def _cache_key(strategy_input: Dict[str, Any]) -> str:
    legs = strategy_input.get("custom_legs") or [{}]
    method = strategy_input.get("validation_method") or strategy_input.get("strike_selection") or "delta"
    key_data = {
        "method": method,
        "symbol": strategy_input.get("symbol"),
        "side": strategy_input.get("side"),
        "dte": strategy_input.get("dte"),
        "delta": strategy_input.get("delta"),
        "quantity": strategy_input.get("quantity"),
        "start": strategy_input.get("options_backtest_start_date"),
        "end": strategy_input.get("options_backtest_end_date"),
        "price_effect": strategy_input.get("price_effect"),
        "legs": legs,
    }
    raw = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _run_cached_tastytrade_strategy(strategy_input: Dict[str, Any]) -> Dict[str, Any]:
    key = _cache_key(strategy_input)
    ttl_seconds = max(1, config.option_validation_cache_ttl_hours) * 3600
    cached = (_read_cache().get(key) or {})
    if cached and time.time() - float(cached.get("created_at", 0)) <= ttl_seconds:
        result = deepcopy(cached.get("result") or {})
        result["cache_hit"] = True
        return result

    result = _run_tastytrade_strategy_bounded(strategy_input)
    if not _is_rate_limited_result(result):
        try:
            _store_cache_entry(key, result)
        except Exception:
            logging.getLogger(__name__).exception(
                "Could not persist options-validation cache entry; validation result remains usable."
            )
    result["cache_hit"] = False
    return result


def _rate_limited_result(strategy_input: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": "VALIDATION_RATE_LIMITED",
        "decision": "REVIEW",
        "strategy_input": strategy_input,
        "backtest": result,
        "error": (
            "Options validation is temporarily busy because Tastytrade returned HTTP 429. "
            "The signal was parsed safely, but no paper option order will be placed until validation is available."
        ),
        "exact_strike_attempt": _attempt_summary(result)
        if strategy_input.get("strike_selection") == "strike"
        else {},
        "delta_proxy_attempt": _attempt_summary(result)
        if strategy_input.get("strike_selection") != "strike"
        else {},
    }


def _complete_exact_strike_fallback_issues(
    option: ParsedOptionSignal, strategy_input: Dict[str, Any]
) -> list[str]:
    """Return issues that prevent safe exact-contract fallback approval."""
    issues: list[str] = []
    if strategy_input.get("strike_selection") != "strike":
        issues.append("not an exact-strike signal")
    if option.strike is None or float(option.strike or 0) <= 0:
        issues.append("missing valid strike")
    if not option.expiry_date and _as_int(strategy_input.get("dte"), 0) <= 0:
        issues.append("missing expiry/DTE")
    if not option.fill_price or float(option.fill_price or 0) <= 0:
        issues.append("missing entry premium")
    if option.fill_price and option.stop_loss is not None and option.stop_loss >= option.fill_price:
        issues.append("stop loss must be below entry premium")
    if option.fill_price and option.target_price is not None and option.target_price <= option.fill_price:
        issues.append("target must be above entry premium")
    if option.fill_price and option.stop_loss is not None and option.target_price is not None:
        risk = float(option.fill_price) - float(option.stop_loss)
        reward = float(option.target_price) - float(option.fill_price)
        rr = reward / risk if risk > 0 else 0.0
        if rr < config.min_option_risk_reward:
            issues.append(
                f"risk/reward {rr:.2f} below minimum {config.min_option_risk_reward:.2f}"
            )
    return issues


def _fallback_option_gate(option: ParsedOptionSignal, strategy_input: Dict[str, Any], reason: str) -> Dict[str, Any]:
    """Approve complete single-leg option signals when paid validation has no data.

    This is not a replacement for the paid backtest. It is a continuity gate for
    Discord signal rooms where the external options backtester can return an
    empty completed result for a valid contract setup.
    """
    issues = []
    if strategy_input.get("strike_selection") == "strike" and (
        option.strike is None or float(option.strike) <= 0
    ):
        issues.append("missing valid strike")
    if str(option.side).upper() not in {"CALL", "PUT"}:
        issues.append("missing CALL/PUT side")
    if _as_int(strategy_input.get("quantity"), 0) <= 0:
        issues.append("missing valid quantity")
    if _as_int(strategy_input.get("dte"), 0) <= 0:
        issues.append("missing valid DTE/expiry")

    if issues:
        return {
            "status": "FALLBACK_REVIEW",
            "decision": "REVIEW",
            "strategy_input": strategy_input,
            "error": f"Paid options validation returned empty data; fallback blocked: {', '.join(issues)}.",
            "fallback_reason": reason,
        }

    exact_fallback_issues = _complete_exact_strike_fallback_issues(option, strategy_input)
    if not exact_fallback_issues:
        return {
            "status": "FALLBACK_APPROVED",
            "decision": "BUY",
            "strategy_input": strategy_input,
            "backtest": {
                "profit_loss": "-",
                "win_rate": "-",
                "note": (
                    "Exact-strike historical providers returned empty data; complete exact-contract "
                    "signal fallback gate used. Alpaca contract verification is still required before order placement."
                ),
            },
            "error": (
                "Exact-strike validation returned empty/all-zero data, but the signal has strike, expiry, "
                "entry premium, SL, target, and acceptable risk/reward. Continuing with exact-contract fallback."
            ),
            "fallback_reason": reason,
            "fallback_approval": "complete_exact_contract_signal",
        }

    if not config.option_allow_unvalidated_fallback:
        return {
            "status": "FALLBACK_REVIEW",
            "decision": "REVIEW",
            "strategy_input": strategy_input,
            "backtest": {
                "profit_loss": "-",
                "win_rate": "-",
                "note": "Paid options backtest returned empty data; exact-contract fallback checks did not pass.",
            },
            "error": (
                "Exact-strike options validation returned empty/all-zero data. "
                f"No option order will be placed because fallback checks failed: {', '.join(exact_fallback_issues)}."
            ),
            "fallback_reason": reason,
            "fallback_issues": exact_fallback_issues,
        }

    return {
        "status": "FALLBACK_APPROVED",
        "decision": "BUY",
        "strategy_input": strategy_input,
        "backtest": {
            "profit_loss": "-",
            "win_rate": "-",
            "note": "Paid options backtest returned empty data; configured unvalidated fallback gate used.",
        },
        "error": (
            "Paid options validation returned empty/all-zero data. "
            "Unvalidated fallback was explicitly enabled; Alpaca contract lookup is still required."
        ),
        "fallback_reason": reason,
        "fallback_approval": "configured_unvalidated_fallback",
    }


def _multi_leg_fallback_gate(
    option: ParsedOptionSignal, strategy_input: Dict[str, Any], reason: str
) -> Dict[str, Any]:
    """Validate a complete multi-leg paper setup when historical trials are empty."""
    assessment = validate_multi_leg_strategy(option)
    issues = list(assessment.get("issues") or [])
    decision = str(assessment.get("decision") or "REVIEW")

    if issues:
        return {
            "status": "MULTI_LEG_FALLBACK_REVIEW",
            "decision": "REVIEW",
            "strategy_input": strategy_input,
            "backtest": {"profit_loss": "-", "win_rate": "-"},
            "error": (
                "Historical multi-leg validation returned no usable trials and structural "
                f"approval was blocked: {', '.join(issues)}."
            ),
            "fallback_reason": reason,
            "fallback_issues": issues,
        }

    return {
        "status": "MULTI_LEG_STRUCTURAL_APPROVED",
        "decision": decision,
        "strategy_input": strategy_input,
        "backtest": {
            "profit_loss": "-",
            "win_rate": "-",
            "note": (
                "Historical multi-leg trials were unavailable. The complete 2-4 leg structure "
                "passed local validation; every exact Alpaca contract and account permission "
                "must still pass before a paper order can be submitted."
            ),
        },
        "error": (
            "Historical multi-leg validation returned no usable trials. Paper-order preparation "
            "continued using complete strategy structure, signal quality, exact-contract, and "
            "account-permission checks."
        ),
        "fallback_reason": reason,
        "fallback_approval": "complete_multi_leg_structure",
        "structural_validation": assessment,
    }


def _run_tastytrade_strategy(strategy_input: Dict[str, Any]) -> Dict[str, Any]:
    custom_legs = list(strategy_input.get("custom_legs") or [])
    return run_options_backtest(
        symbol=strategy_input["symbol"],
        start_date=strategy_input["options_backtest_start_date"],
        end_date=strategy_input["options_backtest_end_date"],
        dte=strategy_input["dte"],
        delta=strategy_input["delta"],
        quantity=strategy_input["quantity"],
        num_legs=max(1, int(strategy_input.get("legs") or len(custom_legs) or 1)),
        custom_legs=custom_legs,
    )


def _run_tastytrade_strategy_bounded(strategy_input: Dict[str, Any]) -> Dict[str, Any]:
    timeout_seconds = max(5, int(config.option_validation_timeout_seconds))
    future = _TASTYTRADE_EXECUTOR.submit(_run_tastytrade_strategy, strategy_input)
    try:
        return future.result(timeout=timeout_seconds)
    except TimeoutError:
        future.cancel()
        return {
            "status": "VALIDATION_TIMEOUT",
            "message": (
                "Options validation took too long, so this signal was moved to review "
                "and the queue continued processing the next signal."
            ),
            "passed_validation": False,
        }


def _attempt_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": result.get("status", "UNKNOWN"),
        "message": result.get("message") or result.get("error") or "",
        "profit_loss": result.get("profit_loss", result.get("total_profit_loss", result.get("pnl", ""))),
        "win_rate": result.get("win_rate", ""),
        "num_trials": result.get("num_trials", ""),
        "backtest_id": result.get("backtest_id", ""),
    }


def _polygon_attempt_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": result.get("status", "UNKNOWN"),
        "message": result.get("message") or result.get("error") or "",
        "profit_loss": result.get("profit_loss", ""),
        "win_rate": result.get("win_rate", ""),
        "num_trials": result.get("bars", ""),
        "polygon_ticker": result.get("polygon_ticker", ""),
        "return_pct": result.get("return_pct", ""),
    }


def _polygon_backtest_result(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": result.get("status"),
        "profit_loss": result.get("profit_loss", "-"),
        "win_rate": result.get("win_rate", "-"),
        "num_trials": result.get("bars", "-"),
        "return_pct": result.get("return_pct", "-"),
        "entry_price": result.get("entry_price", "-"),
        "exit_price": result.get("exit_price", "-"),
        "polygon_ticker": result.get("polygon_ticker", "-"),
        "message": result.get("message", ""),
    }


def _run_polygon_exact_strike_validation(
    option: ParsedOptionSignal,
    strategy_input: Dict[str, Any],
) -> Dict[str, Any]:
    result = validate_exact_strike_with_polygon(
        option,
        strategy_input["options_backtest_start_date"],
        strategy_input["options_backtest_end_date"],
    )
    if result.get("status") == "SUCCESS":
        return {
            "status": "SUCCESS_POLYGON_STRIKE",
            "decision": str(result.get("decision") or "REVIEW").upper(),
            "strategy_input": {
                **strategy_input,
                "validation_method": "polygon_exact_strike",
                "validation_note": (
                    "Exact strike signal validated using Polygon historical option "
                    "aggregates for the real strike/expiry contract."
                ),
            },
            "backtest": _polygon_backtest_result(result),
            "polygon_strike_attempt": _polygon_attempt_summary(result),
            "exact_strike_attempt": {
                **_polygon_attempt_summary(result),
                "status": "SUCCESS_POLYGON",
            },
            "error": "Polygon exact-strike historical validation succeeded.",
        }
    return {
        "status": "POLYGON_STRIKE_UNAVAILABLE",
        "decision": "REVIEW",
        "strategy_input": {
            **strategy_input,
            "validation_method": "polygon_exact_strike",
        },
        "backtest": _polygon_backtest_result(result),
        "polygon_strike_attempt": _polygon_attempt_summary(result),
        "exact_strike_attempt": {
            **_polygon_attempt_summary(result),
            "status": result.get("status", "UNKNOWN"),
        },
        "error": result.get("message") or "Polygon exact-strike validation unavailable.",
    }


def _attach_polygon_attempt(result: Dict[str, Any], polygon_validation: Dict[str, Any] | None) -> Dict[str, Any]:
    if not polygon_validation:
        return result
    attached = dict(result)
    attached.setdefault("polygon_strike_attempt", polygon_validation.get("polygon_strike_attempt") or {})
    if polygon_validation.get("error"):
        attached.setdefault("polygon_error", polygon_validation.get("error"))
    return attached


def _delta_proxy_strategy_input(strategy_input: Dict[str, Any]) -> Dict[str, Any]:
    retry_input = deepcopy(strategy_input)
    retry_input["strike_selection"] = "delta"
    retry_input["strike_price"] = None
    retry_input["requested_strike"] = strategy_input.get("requested_strike")
    retry_input["validation_method"] = "delta_proxy_after_exact_strike_empty"
    retry_input["validation_note"] = (
        "Exact fixed-strike historical validation returned no trials, so the "
        "backtester retried with the same side/DTE/quantity using delta selection. "
        "Alpaca order lookup still uses the exact strike from the signal."
    )
    retry_legs = []
    for leg in retry_input.get("custom_legs") or []:
        retry_leg = deepcopy(leg)
        retry_leg["strikeSelection"] = "delta"
        retry_leg.pop("strikePrice", None)
        retry_legs.append(retry_leg)
    retry_input["custom_legs"] = retry_legs
    return retry_input


def _recent_exact_strike_strategy_input(strategy_input: Dict[str, Any]) -> Dict[str, Any]:
    retry_input = deepcopy(strategy_input)
    retry_days = max(1, int(config.option_exact_strike_recent_retry_days))
    try:
        end_day = datetime.strptime(
            str(strategy_input["options_backtest_end_date"])[:10], "%Y-%m-%d"
        ).date()
    except Exception:
        end_day = date.today()
    start_day = end_day - timedelta(days=retry_days)
    retry_input["options_backtest_start_date"] = start_day.strftime("%Y-%m-%d")
    retry_input["validation_method"] = "exact_strike_recent_window_retry"
    retry_input["validation_note"] = (
        "Full-window exact fixed-strike validation returned no usable trials, so the "
        f"backtester retried exact strike over the most recent {retry_days} days. "
        "Alpaca order lookup still uses the exact strike and expiry from the signal."
    )
    return retry_input


def _delta_proxy_primary_strategy_input(strategy_input: Dict[str, Any]) -> Dict[str, Any]:
    retry_input = _delta_proxy_strategy_input(strategy_input)
    retry_input["validation_method"] = "delta_proxy_primary_for_exact_strike_signal"
    retry_input["validation_note"] = (
        "Tastytrade fixed-strike historical backtests returned unusable all-zero data in testing. "
        "Validation used delta/DTE proxy directly to avoid unnecessary backtester calls. "
        "Alpaca order lookup still uses the exact strike and expiry from the signal."
    )
    return retry_input


def run_options_strategy_validation(option: ParsedOptionSignal) -> Dict[str, Any]:
    """Run project options validation/backtest and return a decision-ready result."""
    strategy_input = build_options_strategy_input(option)
    if option.is_multi_leg:
        structural = validate_multi_leg_strategy(option)
        if not structural["passed"]:
            return {
                "status": "MULTI_LEG_FALLBACK_REVIEW",
                "decision": "REVIEW",
                "strategy_input": strategy_input,
                "backtest": {"profit_loss": "-", "win_rate": "-"},
                "structural_validation": structural,
                "error": "Multi-leg preflight blocked the strategy: " + "; ".join(structural["issues"]),
            }
    polygon_validation: Dict[str, Any] | None = None
    if (
        strategy_input.get("strike_selection") == "strike"
        and not option.is_multi_leg
        and config.option_strike_validation_provider in {"polygon", "polygon_first"}
    ):
        polygon_validation = _run_polygon_exact_strike_validation(option, strategy_input)
        if polygon_validation.get("status") == "SUCCESS_POLYGON_STRIKE":
            return polygon_validation
        if config.option_strike_validation_provider == "polygon":
            return polygon_validation

    if run_options_backtest is None:
        return _attach_polygon_attempt({
            "status": "FAILED",
            "decision": "REVIEW",
            "strategy_input": strategy_input,
            "error": f"Options strategy validation unavailable: {_IMPORT_ERROR}",
        }, polygon_validation)

    try:
        result = _run_cached_tastytrade_strategy(strategy_input)
    except Exception as exc:
        return _attach_polygon_attempt({
            "status": "FAILED",
            "decision": "REVIEW",
            "strategy_input": strategy_input,
            "error": f"Options strategy validation error: {type(exc).__name__}: {exc}",
        }, polygon_validation)
    if _is_rate_limited_result(result):
        return _attach_polygon_attempt(_rate_limited_result(strategy_input, result), polygon_validation)

    decision = _decision_from_backtest(result)
    if _is_empty_backtest_result(result):
        if strategy_input.get("strike_selection") == "strike":
            recent_input = _recent_exact_strike_strategy_input(strategy_input)
            try:
                recent_result = _run_cached_tastytrade_strategy(recent_input)
            except Exception as exc:
                recent_result = {
                    "status": "ERROR",
                    "message": f"Recent exact-strike retry error: {type(exc).__name__}: {exc}",
                }
            if _is_rate_limited_result(recent_result):
                rate_result = _rate_limited_result(recent_input, recent_result)
                rate_result["exact_strike_attempt"] = _attempt_summary(result)
                rate_result["recent_exact_strike_attempt"] = _attempt_summary(recent_result)
                return _attach_polygon_attempt(rate_result, polygon_validation)
            recent_decision = _decision_from_backtest(recent_result)
            if recent_result.get("status") == "SUCCESS":
                return _attach_polygon_attempt({
                    "status": "SUCCESS_EXACT_STRIKE_RECENT",
                    "decision": recent_decision,
                    "strategy_input": recent_input,
                    "backtest": recent_result,
                    "primary_backtest": result,
                    "exact_strike_attempt": _attempt_summary(result),
                    "recent_exact_strike_attempt": _attempt_summary(recent_result),
                    "error": recent_input["validation_note"],
                }, polygon_validation)

            result = {
                **result,
                "recent_retry_status": recent_result.get("status"),
                "recent_retry_message": recent_result.get("message") or recent_result.get("error") or "",
            }
        if option.is_multi_leg:
            fallback_result = _multi_leg_fallback_gate(
                option,
                strategy_input,
                result.get("message")
                or result.get("error")
                or "empty multi-leg options backtest result",
            )
            fallback_result["exact_strike_attempt"] = _attempt_summary(result)
            if result.get("recent_retry_status"):
                fallback_result["recent_exact_strike_attempt"] = {
                    "status": result.get("recent_retry_status"),
                    "message": result.get("recent_retry_message", ""),
                }
            return _attach_polygon_attempt(fallback_result, polygon_validation)
        fallback_result = _fallback_option_gate(
            option,
            strategy_input,
            result.get("message") or result.get("error") or "empty options backtest result",
        )
        fallback_result["exact_strike_attempt"] = _attempt_summary(result)
        if result.get("recent_retry_status"):
            fallback_result["recent_exact_strike_attempt"] = {
                "status": result.get("recent_retry_status"),
                "message": result.get("recent_retry_message", ""),
            }
        fallback_result["delta_proxy_attempt"] = {}
        return _attach_polygon_attempt(fallback_result, polygon_validation)

    return _attach_polygon_attempt({
        "status": result.get("status", "UNKNOWN"),
        "decision": decision,
        "strategy_input": strategy_input,
        "backtest": result,
        "exact_strike_attempt": _attempt_summary(result) if strategy_input.get("strike_selection") == "strike" else {},
        "error": result.get("message") or result.get("error") or "",
    }, polygon_validation)
