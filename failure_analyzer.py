"""
failure_analyzer.py — Batch failure analysis for AI Financial Analyst System.

Reads batch_validation_results[_<tag>].csv and classifies every miss into a
structured failure category with recommended prompt/model fixes.

Usage:
    python failure_analyzer.py
    python failure_analyzer.py --symbol SPY TSLA
    python failure_analyzer.py --horizon 7 14
    python failure_analyzer.py --tag baseline
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "data" / "batch_validation_results.csv"
OUTPUT_JSON = ROOT / "data" / "failure_analysis.json"

FAILURE_CATEGORIES = {
    "OVER_HOLD":                 "AI called HOLD when market moved strongly.",
    "FALSE_BUY":                 "AI called BUY but actual was SELL or flat.",
    "FALSE_SELL":                "AI called SELL but actual was BUY.",
    "WRONG_MAGNITUDE":           "Direction correct but return magnitude badly off.",
    "TREND_REVERSAL_MISSED":     "AI followed trend that reversed sharply.",
    "VOLATILITY_SHIFT_MISSED":   "Market vol regime changed; prompt lacked vol context.",
    "BENCHMARK_CONTEXT_MISSING": "Missing relative-strength vs benchmark.",
    "NEWS_EARNINGS_MISSING":     "Earnings/news event not captured in AI context.",
    "MOMENTUM_CONFLICT_IGNORED": "RSI overbought/oversold conflicted with AI direction.",
    "SUPPORT_RESISTANCE_MISREAD": "Near support/resistance but AI called wrong direction.",
    "EXACT_STRIKE_UNAVAILABLE":  "Options exact strike unsupported by provider.",
    "STRIKE_DTE_MISMATCH":       "Options DTE / strike mismatch with market.",
    "DATA_QUALITY_ISSUE":        "Historical bars missing or inconsistent.",
    "DATA_UNAVAILABLE":          "Historical data provider failed completely.",
    "UNKNOWN_FAILURE":           "Unclassified miss — needs manual review.",
}

RECOMMENDED_FIXES = {
    "OVER_HOLD":                 "Lower HOLD threshold; add momentum confirmation gate.",
    "FALSE_BUY":                 "Add bearish-divergence check; raise buy confidence threshold.",
    "FALSE_SELL":                "Add support-level check; raise sell confidence threshold.",
    "WRONG_MAGNITUDE":           "Add volatility-adjusted return forecast to Gemini prompt.",
    "TREND_REVERSAL_MISSED":     "Add reversal indicator (RSI divergence, volume spike).",
    "VOLATILITY_SHIFT_MISSED":   "Include rolling vol / ATR in feature packet.",
    "BENCHMARK_CONTEXT_MISSING": "Include SPY/QQQ relative performance in prompt.",
    "NEWS_EARNINGS_MISSING":     "Pre-check earnings calendar; warn model if earnings in window.",
    "MOMENTUM_CONFLICT_IGNORED": "Add RSI overbought/oversold gate to BUY/SELL decision.",
    "SUPPORT_RESISTANCE_MISREAD":"Weight near-support/resistance signals more heavily in prompt.",
    "EXACT_STRIKE_UNAVAILABLE":  "Use delta mode for backtesting; note exact-strike limitation.",
    "STRIKE_DTE_MISMATCH":       "Validate DTE and strike against live chain before order.",
    "DATA_QUALITY_ISSUE":        "Re-run with verified historical data; skip bad bars.",
    "DATA_UNAVAILABLE":          "Configure reliable historical provider; retry later.",
    "UNKNOWN_FAILURE":           "Manual review — log to calibration dataset.",
}


def classify_failure(row: dict) -> str:
    """Return failure category string for a single batch CSV row."""
    fr = (row.get("failure_reason") or "").upper()
    if fr == "DATA_UNAVAILABLE":
        return "DATA_UNAVAILABLE"
    if "DATA_QUALITY" in fr:
        return "DATA_QUALITY_ISSUE"
    if fr == "NONE" or row.get("agreement") == "MATCH":
        return "NONE"

    ai_dec  = (row.get("ai_decision") or "").upper()
    act_dec = (row.get("actual_decision") or "").upper()

    if ai_dec == "HOLD" and act_dec in ("BUY", "SELL"):
        return "OVER_HOLD"
    if ai_dec == "BUY" and act_dec == "SELL":
        return "FALSE_BUY"
    if ai_dec == "SELL" and act_dec == "BUY":
        return "FALSE_SELL"

    pred_ret = _safe_float(row.get("predicted_return_pct"))
    act_ret  = _safe_float(row.get("actual_return_pct"))
    ret_err  = _safe_float(row.get("return_error_pp"))
    if pred_ret is not None and act_ret is not None:
        if (pred_ret * act_ret > 0) and abs(ret_err or 0) > 3.0:
            return "WRONG_MAGNITUDE"

    for cat in FAILURE_CATEGORIES:
        if cat in fr:
            return cat

    return "UNKNOWN_FAILURE"


def analyze_hold_bias(rows: List[dict]) -> dict:
    """Calculate HOLD bias metrics."""
    valid = [r for r in rows if r.get("validation_status") not in ("ERROR", "DATA_UNAVAILABLE", None)]
    n = len(valid)
    if n == 0:
        return {"error": "No valid rows"}

    holds = [r for r in valid if (r.get("ai_decision") or "").upper() == "HOLD"]
    hold_rate = round(len(holds) / n * 100, 1)

    # False neutral: AI=HOLD but actual was BUY or SELL with abs return > 2%
    false_neutral = []
    for r in holds:
        act = (r.get("actual_decision") or "").upper()
        if act in ("BUY", "SELL"):
            act_ret = _safe_float(r.get("actual_return_pct"))
            if act_ret is not None and abs(act_ret) > 2.0:
                false_neutral.append(r)

    false_neutral_rate = round(len(false_neutral) / max(len(holds), 1) * 100, 1)
    avg_abs_ret_when_hold = None
    if holds:
        vals = [abs(_safe_float(r.get("actual_return_pct")) or 0) for r in holds]
        avg_abs_ret_when_hold = round(sum(vals) / len(vals), 2)

    # By symbol
    hold_by_symbol = {}
    for sym in sorted({r.get("symbol", "") for r in valid if r.get("symbol")}):
        sym_rows = [r for r in valid if r.get("symbol") == sym]
        sym_holds = [r for r in sym_rows if (r.get("ai_decision") or "").upper() == "HOLD"]
        hold_by_symbol[sym] = round(len(sym_holds) / max(len(sym_rows), 1) * 100, 1)

    # By horizon
    hold_by_horizon = {}
    for h in sorted(set(_safe_int(r.get("horizon_days")) for r in valid) - {None}):
        h_rows = [r for r in valid if _safe_int(r.get("horizon_days")) == h]
        h_holds = [r for r in h_rows if (r.get("ai_decision") or "").upper() == "HOLD"]
        hold_by_horizon[str(h)] = round(len(h_holds) / max(len(h_rows), 1) * 100, 1)

    top_overheld_symbols = sorted(hold_by_symbol.items(), key=lambda x: -x[1])[:5]

    severity = "OK"
    if hold_rate > 55:
        severity = "CRITICAL"
    elif hold_rate > 45:
        severity = "CAUTION"

    return {
        "overall_hold_rate": hold_rate,
        "severity": severity,
        "hold_count": len(holds),
        "false_neutral_count": len(false_neutral),
        "false_neutral_rate": false_neutral_rate,
        "avg_abs_actual_return_when_ai_hold": avg_abs_ret_when_hold,
        "hold_rate_by_symbol": hold_by_symbol,
        "hold_rate_by_horizon": hold_by_horizon,
        "top_overheld_symbols": [{"symbol": s, "hold_rate": r} for s, r in top_overheld_symbols],
        "warning": (
            "MODEL OVER-CONSERVATIVE — HOLD bias is reducing decision match."
            if hold_rate > 50 else ""
        ),
    }


def analyze(
    csv_path: Path = DEFAULT_CSV,
    symbols: Optional[List[str]] = None,
    horizons: Optional[List[int]] = None,
) -> dict:
    """Read CSV, classify failures, return structured analysis dict."""
    if not csv_path.exists():
        return {"error": f"CSV not found: {csv_path}", "rows_analyzed": 0}

    rows = _read_csv(csv_path)
    if symbols:
        rows = [r for r in rows if r.get("symbol") in symbols]
    if horizons:
        rows = [r for r in rows if _safe_int(r.get("horizon_days")) in horizons]

    total        = len(rows)
    data_unavail = [r for r in rows if r.get("validation_status") == "DATA_UNAVAILABLE"]
    valid        = [r for r in rows if r.get("validation_status") not in ("ERROR", "DATA_UNAVAILABLE", None)]

    categories: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        cat = classify_failure(r)
        categories[cat].append(r)

    n_valid   = len(valid)
    n_matches = len([r for r in valid if r.get("agreement") == "MATCH"])

    by_symbol: Dict[str, dict] = {}
    for sym in sorted({r.get("symbol", "") for r in rows if r.get("symbol")}):
        sym_rows  = [r for r in valid if r.get("symbol") == sym]
        sym_match = [r for r in sym_rows if r.get("agreement") == "MATCH"]
        sym_cats: Dict[str, int] = defaultdict(int)
        for r in sym_rows:
            if r.get("agreement") != "MATCH":
                sym_cats[classify_failure(r)] += 1
        by_symbol[sym] = {
            "valid_runs": len(sym_rows),
            "matches":    len(sym_match),
            "match_pct":  round(len(sym_match) / len(sym_rows) * 100, 1) if sym_rows else 0.0,
            "top_failures": dict(sorted(sym_cats.items(), key=lambda x: -x[1])[:5]),
        }

    by_horizon: Dict[str, dict] = {}
    for h in sorted(set(_safe_int(r.get("horizon_days")) for r in rows) - {None}):
        h_rows  = [r for r in valid if _safe_int(r.get("horizon_days")) == h]
        h_match = [r for r in h_rows if r.get("agreement") == "MATCH"]
        by_horizon[str(h)] = {
            "valid_runs": len(h_rows),
            "matches":    len(h_match),
            "match_pct":  round(len(h_match) / len(h_rows) * 100, 1) if h_rows else 0.0,
        }

    top_failures = [
        {
            "category":    cat,
            "count":       len(recs),
            "description": FAILURE_CATEGORIES.get(cat, ""),
            "fix":         RECOMMENDED_FIXES.get(cat, ""),
        }
        for cat, recs in sorted(categories.items(), key=lambda x: -len(x[1]))
        if cat != "NONE"
    ]

    result = {
        "csv_analyzed":           str(csv_path),
        "rows_analyzed":          total,
        "valid_runs":             n_valid,
        "data_unavailable_runs":  len(data_unavail),
        "matches":                n_matches,
        "mismatches":             n_valid - n_matches,
        "match_pct":              round(n_matches / n_valid * 100, 1) if n_valid else 0.0,
        "top_failure_categories": top_failures,
        "by_symbol":              by_symbol,
        "by_horizon":             by_horizon,
        "symbols_with_data_unavailable": list({r.get("symbol") for r in data_unavail}),
    }

    # Include HOLD bias analysis
    result["hold_bias"] = analyze_hold_bias(rows)

    return result


def _read_csv(path: Path) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _safe_float(v) -> Optional[float]:
    try:
        return float(v) if v not in (None, "", "None") else None
    except (TypeError, ValueError):
        return None


def _safe_int(v) -> Optional[int]:
    try:
        return int(v) if v not in (None, "", "None") else None
    except (TypeError, ValueError):
        return None


def _print_analysis(result: dict) -> None:
    print(f"\n-- Failure Analysis: {result.get('csv_analyzed', '?')} --")
    if result.get("error"):
        print(f"  ERROR: {result['error']}")
        return
    print(f"  Rows analyzed:       {result['rows_analyzed']}")
    print(f"  Valid runs:          {result['valid_runs']}")
    print(f"  Data unavailable:    {result['data_unavailable_runs']}")
    print(f"  Match %:             {result['match_pct']}%")
    print("\n  Top failure categories:")
    for item in result["top_failure_categories"][:8]:
        print(f"    {item['count']:4d}x  {item['category']}")
        print(f"           Fix: {item['fix']}")
    print("\n  Per-symbol match %:")
    for sym, d in sorted(result["by_symbol"].items()):
        print(f"    {sym:8s}  {d['match_pct']:5.1f}%  ({d['matches']}/{d['valid_runs']} runs)")
    if result.get("symbols_with_data_unavailable"):
        print(f"\n  Symbols with data unavailable: {result['symbols_with_data_unavailable']}")

    # Print HOLD bias section
    hb = result.get("hold_bias", {})
    if hb and not hb.get("error"):
        print(f"\n  -- HOLD Bias Analysis --")
        print(f"    Overall hold rate:       {hb.get('overall_hold_rate', '?')}%  [{hb.get('severity', '?')}]")
        print(f"    Hold count:              {hb.get('hold_count', '?')}")
        print(f"    False neutral count:     {hb.get('false_neutral_count', '?')}")
        print(f"    False neutral rate:      {hb.get('false_neutral_rate', '?')}%")
        print(f"    Avg abs ret when HOLD:   {hb.get('avg_abs_actual_return_when_ai_hold', '?')}%")
        if hb.get("warning"):
            print(f"    WARNING: {hb['warning']}")
        if hb.get("top_overheld_symbols"):
            print(f"    Top over-held symbols:   {hb['top_overheld_symbols']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Failure Analyzer")
    parser.add_argument("--csv",     default="")
    parser.add_argument("--symbol",  nargs="*")
    parser.add_argument("--horizon", nargs="*", type=int)
    parser.add_argument("--out",     default="")
    parser.add_argument("--tag",     default="",
                        help="Tag for input/output file naming (e.g. 'baseline')")
    args = parser.parse_args()

    # Resolve paths based on tag
    if args.tag:
        csv_path = ROOT / "data" / f"batch_validation_results_{args.tag}.csv"
        out_path = ROOT / "data" / f"failure_analysis_{args.tag}.json"
    else:
        csv_path = Path(args.csv) if args.csv else DEFAULT_CSV
        out_path = Path(args.out) if args.out else OUTPUT_JSON

    result = analyze(csv_path, symbols=args.symbol, horizons=args.horizon)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    _print_analysis(result)
    print(f"\n  Full JSON: {out_path}")
