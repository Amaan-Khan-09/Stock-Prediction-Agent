"""Bridge from Discord signals to the project stock prediction pipeline."""
from __future__ import annotations

import os
import json
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

from .config import config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(1, str(PROJECT_ROOT / "tools"))
load_dotenv(PROJECT_ROOT / ".env", override=False)

from historical_price_service import fetch_price_history, get_provider_used  # noqa: E402
from stock_prediction_agent import (  # noqa: E402
    build_stock_prediction_hash,
    run_stock_prediction,
    validate_stock_prediction_input,
)
from stock_walkforward_validator import run_stock_validation  # noqa: E402

try:
    from gemini_stock_prediction_agent import run_gemini_stock_prediction  # noqa: E402
except Exception:  # pragma: no cover - optional provider
    run_gemini_stock_prediction = None


def _project_venv_python() -> Path:
    return PROJECT_ROOT / "venv" / "Scripts" / "python.exe"


def _should_delegate_to_project_venv() -> bool:
    if os.getenv("DISCORD_AGENT_SKIP_VENV_DELEGATE") == "1":
        return False
    venv_python = _project_venv_python()
    if not venv_python.exists():
        return False
    try:
        return Path(sys.executable).resolve() != venv_python.resolve()
    except Exception:
        return True


def _run_prediction_in_project_venv(symbol: str, horizon_days: int | None = None) -> Dict[str, Any]:
    code = (
        "import json; "
        "from discord_stock_prediction_agent.prediction_bridge import run_project_prediction; "
        f"r=run_project_prediction({symbol!r}, {repr(horizon_days)}); "
        "print(json.dumps(r, default=str))"
    )
    env = os.environ.copy()
    env["DISCORD_AGENT_SKIP_VENV_DELEGATE"] = "1"
    proc = subprocess.run(
        [str(_project_venv_python()), "-c", code],
        cwd=str(PROJECT_ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
    )
    if proc.returncode != 0:
        return {
            "status": "FAILED",
            "decision": "REVIEW",
            "symbol": symbol.upper(),
            "error": (proc.stderr or proc.stdout or "Project venv prediction failed.")[-1000:],
        }
    for line in reversed((proc.stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    return {
        "status": "FAILED",
        "decision": "REVIEW",
        "symbol": symbol.upper(),
        "error": "Project venv prediction returned no JSON result.",
    }


def _iso(day: date) -> str:
    return day.strftime("%Y-%m-%d")


def _previous_trading_day(today: date | None = None) -> date:
    day = (today or date.today()) - timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def build_default_prediction_input(symbol: str, horizon_days: int | None = None) -> Dict[str, Any]:
    origin = _previous_trading_day()
    horizon = int(horizon_days or config.default_horizon_days)
    target = date.today()
    if target <= origin:
        target = origin + timedelta(days=max(1, horizon))
    context_start = origin - timedelta(days=config.historical_context_days)
    return {
        "symbol": symbol.upper(),
        "historical_context_start_date": _iso(context_start),
        "prediction_origin_date": _iso(origin),
        "decision_horizon_days": horizon,
        "target_date": _iso(target),
        "initial_capital": config.initial_capital,
        "benchmark": config.benchmark,
        "validation_mode": "horizon_days",
        "price_basis": config.price_basis,
    }


def run_project_prediction(symbol: str, horizon_days: int | None = None) -> Dict[str, Any]:
    """Run one-day walk-forward prediction plus actual historical validation."""
    if _should_delegate_to_project_venv():
        return _run_prediction_in_project_venv(symbol, horizon_days)

    spi = build_default_prediction_input(symbol, horizon_days)
    valid, err = validate_stock_prediction_input(spi)
    input_hash = build_stock_prediction_hash(spi)
    if not valid:
        return {
            "status": "FAILED",
            "decision": "REVIEW",
            "symbol": symbol.upper(),
            "error": f"Input validation failed: {err}",
            "stock_prediction_input": spi,
            "stock_prediction_input_hash": input_hash,
        }

    try:
        ctx_start = datetime.strptime(spi["historical_context_start_date"], "%Y-%m-%d").date()
        min_days = max(400, (date.today() - ctx_start).days + 90)
    except Exception:
        min_days = 500

    hist, hist_err = fetch_price_history(spi["symbol"], min_days=min_days)
    if not hist:
        return {
            "status": "FAILED",
            "decision": "REVIEW",
            "symbol": spi["symbol"],
            "error": hist_err,
            "stock_prediction_input": spi,
            "stock_prediction_input_hash": input_hash,
        }

    ctx_bars = [
        bar for bar in hist
        if spi["historical_context_start_date"] <= bar["date"] <= spi["prediction_origin_date"]
    ]
    if not ctx_bars:
        return {
            "status": "FAILED",
            "decision": "REVIEW",
            "symbol": spi["symbol"],
            "error": "No historical bars available in the context window.",
            "stock_prediction_input": spi,
            "stock_prediction_input_hash": input_hash,
        }

    use_gemini = (
        os.getenv("AI_PROVIDER", "gemini").lower() == "gemini"
        and run_gemini_stock_prediction is not None
        and (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    )
    provider_warning = ""
    if use_gemini:
        ai_result = run_gemini_stock_prediction(spi, ctx_bars)
        provider = "gemini_project_agent"
        if ai_result.get("status") != "SUCCESS":
            provider_warning = ai_result.get("error", "Gemini provider failed.")
            if os.getenv("ALLOW_BASELINE_FALLBACK", "false").lower() == "true":
                ai_result = run_stock_prediction(spi, ctx_bars)
                ai_result["ai_provider"] = "baseline_fallback"
                ai_result["gemini_used"] = False
                ai_result["fallback_reason"] = provider_warning
                provider = "baseline_project_agent_fallback"
            else:
                provider = "gemini_project_agent"
    else:
        ai_result = run_stock_prediction(spi, ctx_bars)
        provider = "baseline_project_agent"

    ai_result["stock_prediction_input"] = spi
    ai_result["stock_prediction_input_hash"] = ai_result.get("stock_prediction_input_hash") or input_hash
    ai_result["historical_price_provider"] = get_provider_used(spi["symbol"])
    ai_result["prediction_provider"] = provider
    if provider_warning:
        ai_result["provider_warning"] = provider_warning

    if ai_result.get("status") != "SUCCESS":
        return {
            "status": "FAILED",
            "decision": "REVIEW",
            "symbol": spi["symbol"],
            "error": ai_result.get("error", "AI prediction failed."),
            "stock_prediction_input": spi,
            "stock_prediction_input_hash": input_hash,
            "ai_prediction": ai_result,
            "actual_validation": {},
        }

    actual_validation = run_stock_validation(spi, ai_result, hist)
    if actual_validation.get("status") != "SUCCESS":
        actual_validation.setdefault("comparison", {"status": actual_validation.get("status", "PENDING")})

    return {
        "status": "SUCCESS",
        "symbol": spi["symbol"],
        "decision": ai_result.get("decision", "REVIEW"),
        "stock_prediction_input": spi,
        "stock_prediction_input_hash": input_hash,
        "ai_prediction": ai_result,
        "actual_validation": actual_validation,
        "comparison": actual_validation.get("comparison", {}),
        "historical_price_provider": get_provider_used(spi["symbol"]),
        "prediction_provider": provider,
        "provider_warning": provider_warning,
    }
