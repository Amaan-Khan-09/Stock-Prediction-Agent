"""Batch Validation Runner — Part L of the implementation spec.

Runs walk-forward stock price validation across multiple symbols and time windows.
Produces data/batch_validation_results[_<tag>].csv and data/batch_validation_summary[_<tag>].json.

Usage:
  python batch_validation_runner.py
  python batch_validation_runner.py --symbols SPY,TSLA --horizons 7,30 --max-runs 20
  python batch_validation_runner.py --focused   (SPY + TSLA only)
  python batch_validation_runner.py --tag baseline

Integrates with existing run_stock_price_validation() — no new API calls invented.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)

DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV  = DATA_DIR / "batch_validation_results.csv"
SUMMARY_JSON = DATA_DIR / "batch_validation_summary.json"

# ── Symbol sets ──────────────────────────────────────────────────────────────

FOCUSED_SYMBOLS = ["SPY", "TSLA"]

TOP_SP500_SYMBOLS = [
    "SPY", "TSLA", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "QQQ", "IWM",
]

EXTENDED_SYMBOLS = TOP_SP500_SYMBOLS + ["BRK-B", "JPM", "UNH"]

DEFAULT_HORIZONS      = [7, 14, 30, 45]
DEFAULT_CTX_DAYS      = 365   # days of context before origin
DEFAULT_MAX_RUNS      = 50
DEFAULT_MONTHS_BACK   = 24    # how many months back to sample origins

CSV_FIELDNAMES = [
    "run_id", "input_hash", "symbol", "mode", "origin_date", "target_date",
    "horizon_days", "price_basis",
    "context_start_date", "bars_visible_to_ai", "last_ai_bar_date",
    "origin_price", "target_price",
    "ai_decision", "actual_decision", "agreement",
    "predicted_return_pct", "actual_return_pct", "return_error_pp",
    "predicted_target_price",
    "confidence", "risk", "bullish_score", "bearish_score", "uncertainty_score",
    "data_quality_score",
    "trend_regime", "volatility_regime",
    "benchmark_return_20d", "relative_strength_20d",
    "validation_status", "exact_validation_status", "proxy_validation_status",
    "validation_type", "accuracy_eligible",
    "failure_reason", "recommended_fix", "provider_status",
]


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def run_batch(
    symbols:      List[str] = FOCUSED_SYMBOLS,
    horizons:     List[int] = DEFAULT_HORIZONS,
    max_runs:     int       = DEFAULT_MAX_RUNS,
    months_back:  int       = DEFAULT_MONTHS_BACK,
    price_basis:  str       = "close",
    mode:         str       = "stock",
    capital:      float     = 50_000.0,
    benchmark:    str       = "SPY",
    verbose:      bool      = True,
    tag:          str       = "",
    out_csv:      Path      = None,
    out_json:     Path      = None,
) -> dict:
    """Run batch validation and return summary dict.

    Writes CSV to data/batch_validation_results[_<tag>].csv and
    JSON summary to data/batch_validation_summary[_<tag>].json.
    """
    import os as _os
    from historical_price_service import fetch_price_history, filter_history_up_to
    from stock_walkforward_validator import run_stock_validation

    _gemini_key = _os.getenv("GEMINI_API_KEY", "") or _os.getenv("GOOGLE_API_KEY", "")
    _ai_provider = _os.getenv("AI_PROVIDER", "gemini").lower()
    _use_gemini  = bool(_gemini_key and _ai_provider == "gemini")

    if _use_gemini:
        from gemini_stock_prediction_agent import run_gemini_stock_prediction as _run_ai
    else:
        from stock_prediction_agent import run_stock_prediction as _run_ai

    # Determine output paths
    if out_csv is None:
        if tag:
            out_csv = DATA_DIR / f"batch_validation_results_{tag}.csv"
        else:
            out_csv = RESULTS_CSV
    if out_json is None:
        if tag:
            out_json = DATA_DIR / f"batch_validation_summary_{tag}.json"
        else:
            out_json = SUMMARY_JSON

    runs_requested  = 0
    runs_completed  = 0
    rows: List[dict] = []

    today       = date.today()
    origins     = _build_origin_dates(months_back, today)

    for symbol in symbols:
        for horizon in horizons:
            for origin in origins:
                if runs_requested >= max_runs:
                    break

                target = origin + timedelta(days=horizon)
                if target >= today:
                    continue  # Can't validate future

                ctx_start = (origin - timedelta(days=DEFAULT_CTX_DAYS)).strftime("%Y-%m-%d")
                origin_s  = origin.strftime("%Y-%m-%d")
                target_s  = target.strftime("%Y-%m-%d")
                run_id    = str(uuid.uuid4())[:8].upper()

                runs_requested += 1
                if verbose:
                    print(f"[{runs_requested:3d}] {symbol} origin={origin_s} horizon={horizon}d ... ", end="", flush=True)

                try:
                    spi = {
                        "symbol":                        symbol,
                        "historical_context_start_date": ctx_start,
                        "prediction_origin_date":        origin_s,
                        "target_date":                   target_s,
                        "decision_horizon_days":         horizon,
                        "initial_capital":               capital,
                        "benchmark":                     benchmark,
                        "validation_mode":               "horizon_days",
                        "price_basis":                   price_basis,
                    }

                    # Fetch price history
                    _days_req = max(400, DEFAULT_CTX_DAYS + 90)
                    hist, hist_err = fetch_price_history(symbol, min_days=_days_req)
                    if not hist:
                        raise ValueError(f"Price data unavailable: {hist_err}")

                    # Filter to [ctx_start, origin] — same window the app uses
                    ctx_bars = [b for b in filter_history_up_to(hist, origin_s)
                                if b.get("date", "") >= ctx_start]
                    if not ctx_bars:
                        raise ValueError(f"No price bars in [{ctx_start}, {origin_s}]")

                    # AI prediction (Gemini when available, else old agent)
                    ai_result = _run_ai(spi, ctx_bars)
                    if ai_result.get("status") != "SUCCESS":
                        raise ValueError(f"AI prediction failed: {ai_result.get('error')}")

                    # Actual validation
                    val_result = run_stock_validation(spi, ai_result, hist)

                    result = {
                        "status":                      val_result.get("status", "FAILED"),
                        "stock_prediction_input_hash": ai_result.get("stock_prediction_input_hash"),
                        "ai_prediction":               ai_result,
                        "actual_validation":           val_result,
                        "comparison":                  val_result.get("comparison", {}),
                    }

                    row = _result_to_row(
                        result, run_id, symbol, mode, origin_s, target_s,
                        horizon, price_basis,
                    )
                    rows.append(row)
                    runs_completed += 1

                    if verbose:
                        match_str = "MATCH" if row.get("agreement") == "MATCH" else "miss"
                        print(f"AI={row.get('ai_decision','?')} actual={row.get('actual_decision','?')} [{match_str}]")

                except Exception as exc:
                    exc_str = str(exc)
                    # Classify data-unavailable errors separately from other errors
                    _is_data_err = any(k in exc_str.lower() for k in (
                        "historical", "data unavailable", "dns", "connection", "timeout",
                        "resolv", "historicalprice", "not available on current plan",
                    ))
                    _fail_reason = "DATA_UNAVAILABLE" if _is_data_err else f"EXCEPTION: {exc_str[:200]}"
                    if verbose:
                        label = "DATA_UNAVAILABLE" if _is_data_err else "ERROR"
                        print(f"{label}: {exc_str[:120]}")
                    rows.append(_error_row(run_id, symbol, mode, origin_s, target_s, horizon,
                                           exc_str, failure_reason=_fail_reason))

    # Write CSV
    _write_csv(rows, out_csv)

    # Build & write summary
    summary = _build_summary(rows, symbols, horizons)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if verbose:
        print(f"\nBatch complete: {runs_completed}/{runs_requested} runs.")
        print(f"CSV:  {out_csv}")
        print(f"JSON: {out_json}")
        _print_summary(summary)

    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_origin_dates(months_back: int, today: date) -> List[date]:
    """Monthly origin dates going back N months."""
    origins = []
    current = (today.replace(day=1) - timedelta(days=1)).replace(day=1)  # start of last month
    for _ in range(months_back):
        origins.append(current)
        current = (current - timedelta(days=1)).replace(day=1)
    return origins


def _result_to_row(result: dict, run_id: str, symbol: str, mode: str,
                   origin_s: str, target_s: str, horizon: int, price_basis: str) -> dict:
    status = result.get("status", "FAILED")
    ai     = result.get("ai_prediction", {})
    val    = result.get("actual_validation", {})
    cmp    = result.get("comparison", {})

    ai_dec     = ai.get("decision")
    actual_dec = val.get("actual_decision")
    agreement  = _compute_agreement(ai_dec, actual_dec)

    pred_ret   = ai.get("predicted_return_pct")
    actual_ret = val.get("actual_return_pct")
    ret_err    = (
        round(float(pred_ret) - float(actual_ret), 4)
        if pred_ret is not None and actual_ret is not None
        else None
    )

    failure_reason, recommended_fix = _classify_failure(
        ai_dec, actual_dec, pred_ret, actual_ret, ai.get("confidence_score"), cmp
    )

    # Signal scores — from ai_prediction or top-level result
    sig_scores = ai.get("signal_scores") or result.get("signal_scores") or {}
    feat = result.get("features_used") or {}
    fp   = result.get("_feature_packet") or {}

    accuracy_eligible = status == "SUCCESS" and agreement in ("MATCH", "MISMATCH")
    provider_status   = "OK" if status == "SUCCESS" else "FAILED"

    return {
        "run_id":          run_id,
        "input_hash":      result.get("stock_prediction_input_hash"),
        "symbol":          symbol,
        "mode":            mode,
        "origin_date":     origin_s,
        "target_date":     target_s,
        "horizon_days":    horizon,
        "price_basis":     price_basis,
        "context_start_date":  result.get("historical_context_start_date") or (
            ai.get("historical_context_start_date") or ""
        ),
        "bars_visible_to_ai":  feat.get("bars_used"),
        "last_ai_bar_date":    feat.get("last_ai_bar_date"),
        "origin_price":        result.get("origin_price_used") or ai.get("origin_price_used"),
        "target_price":        val.get("target_price"),
        "ai_decision":     ai_dec or "",
        "actual_decision": actual_dec or "",
        "agreement":       agreement,
        "predicted_return_pct": pred_ret,
        "actual_return_pct":    actual_ret,
        "return_error_pp":      ret_err,
        "predicted_target_price": ai.get("predicted_target_price"),
        "confidence":      ai.get("confidence_score"),
        "risk":            ai.get("risk_score"),
        "bullish_score":   sig_scores.get("bullish_score"),
        "bearish_score":   sig_scores.get("bearish_score"),
        "uncertainty_score": sig_scores.get("uncertainty_score"),
        "data_quality_score": ai.get("data_quality_score") or result.get("data_quality_score"),
        "trend_regime":    feat.get("trend_regime") or fp.get("trend", {}).get("trend_regime"),
        "volatility_regime": fp.get("volatility", {}).get("volatility_regime"),
        "benchmark_return_20d": fp.get("benchmark_comparison", {}).get("benchmark_return_20d"),
        "relative_strength_20d": fp.get("benchmark_comparison", {}).get("relative_strength_20d"),
        "validation_status":       status,
        "exact_validation_status": "NOT_APPLICABLE",
        "proxy_validation_status": "NOT_APPLICABLE",
        "validation_type":         "stock_price_validation",
        "accuracy_eligible":       accuracy_eligible,
        "failure_reason":  failure_reason,
        "recommended_fix": recommended_fix,
        "provider_status": provider_status,
    }


def _error_row(run_id, symbol, mode, origin_s, target_s, horizon, error_msg,
               failure_reason: str = "") -> dict:
    row = {k: None for k in CSV_FIELDNAMES}
    _fr = failure_reason or f"EXCEPTION: {error_msg[:200]}"
    _status = "DATA_UNAVAILABLE" if _fr == "DATA_UNAVAILABLE" else "ERROR"
    _fix    = "DATA_UNAVAILABLE" if _fr == "DATA_UNAVAILABLE" else "DATA_QUALITY_ISSUE"
    row.update({
        "run_id": run_id, "symbol": symbol, "mode": mode,
        "origin_date": origin_s, "target_date": target_s,
        "horizon_days": horizon, "validation_status": _status,
        "failure_reason": _fr,
        "recommended_fix": _fix,
        "accuracy_eligible": False,
        "provider_status": "FAILED",
    })
    return row


def _compute_agreement(ai_dec: Optional[str], actual_dec: Optional[str]) -> str:
    if ai_dec is None or actual_dec is None:
        return "UNKNOWN"
    if ai_dec.upper() == actual_dec.upper():
        return "MATCH"
    return "MISMATCH"


def _classify_failure(
    ai_dec, actual_dec, pred_ret, actual_ret, confidence, cmp: dict
) -> Tuple[str, str]:
    """Return (failure_reason, recommended_fix) strings."""
    if ai_dec is None or actual_dec is None:
        return "DATA_QUALITY_ISSUE", "DATA_QUALITY_ISSUE"

    ai  = (ai_dec or "").upper()
    act = (actual_dec or "").upper()

    if ai == act:
        return "NONE", "NONE"

    pred_f   = _safe_float(pred_ret) or 0.0
    actual_f = _safe_float(actual_ret) or 0.0
    ret_err  = abs(pred_f - actual_f)

    if ai == "HOLD" and act in ("BUY", "SELL"):
        if ret_err > 10:
            return "VOLATILITY_SHIFT_MISSED", "Widen decision thresholds; add vol-shift detection."
        return "OVER_HOLD", "Widen decision thresholds or add momentum confirmation."

    if ai == "BUY" and act == "SELL":
        if actual_f < -5:
            return "TREND_REVERSAL_MISSED", "Add earnings/news context to prompt."
        if ret_err > 5:
            return "WRONG_MAGNITUDE", "Improve predicted return calibration."
        return "FALSE_BUY", "Reduce buy signal sensitivity or add reversal check."

    if ai == "SELL" and act == "BUY":
        if actual_f > 5:
            return "TREND_REVERSAL_MISSED", "Add RSI oversold check; market reversed from sell signal."
        return "FALSE_SELL", "Check for oversold RSI false-SELL trigger. Add reversal_risk check."

    if ai in ("BUY", "SELL") and act == "HOLD":
        if confidence is not None and _safe_float(confidence) is not None and float(confidence) > 70:
            return "BENCHMARK_CONTEXT_MISSING", "High-confidence wrong direction — include relative-strength vs benchmark."
        return "WRONG_MAGNITUDE", "Improve predicted return calibration."

    return "UNKNOWN_FAILURE", "Review feature packet and prompt reasoning."


def _safe_float(v) -> Optional[float]:
    try:
        return float(v) if v not in (None, "", "None") else None
    except (TypeError, ValueError):
        return None


def _write_csv(rows: List[dict], path: Path = None) -> None:
    if path is None:
        path = RESULTS_CSV
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDNAMES})


def _build_summary(rows: List[dict], symbols: List[str], horizons: List[int]) -> dict:
    data_unavail = [r for r in rows if r.get("validation_status") == "DATA_UNAVAILABLE"]
    failed       = [r for r in rows if r.get("validation_status") == "ERROR"]
    valid        = [r for r in rows if r.get("validation_status") not in ("ERROR", "DATA_UNAVAILABLE", None)]

    n_total = len(rows)
    n_valid = len(valid)
    n_failed = len(failed)
    n_data_unavail = len(data_unavail)
    symbols_failed = list({r.get("symbol", "") for r in failed + data_unavail if r.get("symbol")})

    matches    = [r for r in valid if r.get("agreement") == "MATCH"]
    mismatches = [r for r in valid if r.get("agreement") == "MISMATCH"]
    decision_match_pct = round(len(matches) / n_valid * 100, 1) if n_valid else 0.0

    # Direction match
    dir_match = [r for r in valid
                 if _direction_match(r.get("ai_decision"), r.get("actual_decision"))]
    direction_match_pct = round(len(dir_match) / n_valid * 100, 1) if n_valid else 0.0

    # Return error
    errs = [r["return_error_pp"] for r in valid
            if r.get("return_error_pp") is not None]
    avg_return_error = round(sum(abs(float(e)) for e in errs) / len(errs), 4) if errs else None

    # Hold rate
    holds = [r for r in valid if (r.get("ai_decision") or "").upper() == "HOLD"]
    hold_rate = round(len(holds) / n_valid * 100, 1) if n_valid else 0.0

    false_buy  = [r for r in mismatches if (r.get("ai_decision") or "").upper() == "BUY"]
    false_sell = [r for r in mismatches if (r.get("ai_decision") or "").upper() == "SELL"]
    over_hold  = [r for r in mismatches
                  if (r.get("ai_decision") or "").upper() == "HOLD"
                  and (r.get("actual_decision") or "").upper() in ("BUY", "SELL")]

    # Per-symbol match rate
    match_by_symbol = {}
    for sym in symbols:
        sym_rows  = [r for r in valid if r.get("symbol") == sym]
        sym_match = [r for r in sym_rows if r.get("agreement") == "MATCH"]
        match_by_symbol[sym] = {
            "runs":       len(sym_rows),
            "matches":    len(sym_match),
            "match_pct":  round(len(sym_match) / len(sym_rows) * 100, 1) if sym_rows else 0.0,
        }

    # Per-horizon match rate
    match_by_horizon = {}
    for h in horizons:
        h_rows  = [r for r in valid if r.get("horizon_days") == h or str(r.get("horizon_days")) == str(h)]
        h_match = [r for r in h_rows if r.get("agreement") == "MATCH"]
        match_by_horizon[str(h)] = {
            "runs":      len(h_rows),
            "matches":   len(h_match),
            "match_pct": round(len(h_match) / len(h_rows) * 100, 1) if h_rows else 0.0,
        }

    # Top failure reasons
    failure_counts: Dict[str, int] = {}
    for r in mismatches:
        reason = r.get("failure_reason") or "UNKNOWN"
        failure_counts[reason] = failure_counts.get(reason, 0) + 1
    top_failures = sorted(failure_counts.items(), key=lambda x: -x[1])

    # Recommended prompt changes
    recommended = _build_recommendations(top_failures, hold_rate, decision_match_pct)

    return {
        "generated_at":          datetime.utcnow().isoformat(),
        "symbols":               symbols,
        "horizons":              horizons,
        "total_runs":            n_total,
        "valid_runs":            n_valid,
        "failed_runs":           n_failed,
        "data_unavailable_count": n_data_unavail,
        "symbols_failed":        symbols_failed,
        "decision_match_pct":    decision_match_pct,
        "direction_match_pct":   direction_match_pct,
        "avg_return_error_pp":   avg_return_error,
        "hold_rate_pct":         hold_rate,
        "false_buy_count":       len(false_buy),
        "false_sell_count":      len(false_sell),
        "over_hold_count":       len(over_hold),
        "match_by_symbol":       match_by_symbol,
        "match_by_horizon":      match_by_horizon,
        "top_failure_reasons":   [{"reason": r, "count": c} for r, c in top_failures],
        "recommended_prompt_changes": recommended,
    }


def _direction_match(ai: Optional[str], actual: Optional[str]) -> bool:
    if not ai or not actual:
        return False
    ai_up  = ai.upper()
    act_up = actual.upper()
    # BUY / positive → same direction; SELL / negative → same direction
    pos = {"BUY"}
    neg = {"SELL"}
    if ai_up in pos and act_up in pos:
        return True
    if ai_up in neg and act_up in neg:
        return True
    return ai_up == act_up


def _build_recommendations(top_failures, hold_rate, decision_match_pct) -> List[str]:
    recs = []
    failure_names = [f[0] for f in top_failures]

    if "OVER_HOLD" in failure_names:
        recs.append("OVER_HOLD detected: widen BUY/SELL thresholds or add stronger signal requirement before HOLD.")
    if "VOLATILITY_SHIFT_MISSED" in failure_names:
        recs.append("VOLATILITY_SHIFT_MISSED: add vol-regime change detection to feature packet.")
    if "TREND_REVERSAL_MISSED" in failure_names:
        recs.append("TREND_REVERSAL_MISSED: inject news/earnings context into Gemini prompt.")
    if "FALSE_BUY" in failure_names:
        recs.append("FALSE_BUY: add reversal_risk and RSI overbought guard to buy logic.")
    if "FALSE_SELL" in failure_names:
        recs.append("FALSE_SELL: add RSI oversold check before SELL to prevent false downside calls.")
    if "WRONG_MAGNITUDE" in failure_names:
        recs.append("WRONG_MAGNITUDE: recalibrate predicted_return_pct scaling vs actual by horizon.")
    if "BENCHMARK_CONTEXT_MISSING" in failure_names:
        recs.append("BENCHMARK_CONTEXT_MISSING: ensure relative_strength vs SPY is in feature packet.")
    if hold_rate > 55:
        recs.append(f"HOLD rate is {hold_rate:.0f}% (CRITICAL) — prompt over-selects HOLD. Add confidence requirement for HOLD.")
    elif hold_rate > 45:
        recs.append(f"HOLD rate is {hold_rate:.0f}% (CAUTION) — consider threshold adjustment.")
    if decision_match_pct < 30:
        recs.append("Match rate below 30% — investigate feature quality and prompt clarity.")

    if not recs:
        recs.append("No dominant failure pattern detected. Continue monitoring across more time windows.")
    return recs


def _print_summary(summary: dict) -> None:
    print("\n-- Batch Validation Summary ------------------------------------------")
    print(f"  Total runs:            {summary['total_runs']}")
    print(f"  Valid runs:            {summary['valid_runs']}")
    print(f"  Failed runs:           {summary['failed_runs']}")
    print(f"  Data unavailable:      {summary.get('data_unavailable_count', 0)}")
    print(f"  Symbols failed:        {summary.get('symbols_failed', [])}")
    print(f"  Decision match %:      {summary['decision_match_pct']}  (valid runs only)")
    print(f"  Direction match %:     {summary['direction_match_pct']}  (valid runs only)")
    print(f"  Avg return error pp:   {summary['avg_return_error_pp']}")
    print(f"  Hold rate %:           {summary['hold_rate_pct']}")
    print(f"  Over-hold count:       {summary['over_hold_count']}")
    print(f"  False buy count:       {summary['false_buy_count']}")
    print(f"  False sell count:      {summary['false_sell_count']}")
    if summary.get("top_failure_reasons"):
        print("  Top failure reasons:")
        for item in summary["top_failure_reasons"][:5]:
            print(f"    {item['count']:3d}x  {item['reason']}")
    if summary.get("recommended_prompt_changes"):
        print("  Recommendations:")
        for rec in summary["recommended_prompt_changes"]:
            print(f"    - {rec}")
    print("----------------------------------------------------------------------")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Batch validation runner")
    p.add_argument("--symbols",   default="SPY,TSLA",
                   help="Comma-separated ticker list (default: SPY,TSLA)")
    p.add_argument("--horizons",  default="7,14,30,45",
                   help="Comma-separated horizon days (default: 7,14,30,45)")
    p.add_argument("--max-runs",  type=int, default=50,
                   help="Maximum number of validation runs (default: 50)")
    p.add_argument("--months-back", type=int, default=24,
                   help="How many months back to sample (default: 24)")
    p.add_argument("--capital",   type=float, default=50_000.0)
    p.add_argument("--benchmark", default="SPY")
    p.add_argument("--focused",   action="store_true",
                   help="Use focused symbol set: SPY + TSLA only")
    p.add_argument("--top-sp500", action="store_true",
                   help="Use top S&P 500 symbol set")
    p.add_argument("--quiet",     action="store_true")
    p.add_argument("--tag",       default="",
                   help="Tag for output file naming (e.g. 'baseline' produces batch_validation_results_baseline.csv)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.focused:
        symbols = FOCUSED_SYMBOLS
    elif args.top_sp500:
        symbols = TOP_SP500_SYMBOLS
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]

    run_batch(
        symbols     = symbols,
        horizons    = horizons,
        max_runs    = args.max_runs,
        months_back = args.months_back,
        capital     = args.capital,
        benchmark   = args.benchmark,
        verbose     = not args.quiet,
        tag         = args.tag,
    )
