"""
calibration_profile_builder.py — Build per-symbol calibration profiles
for the AI Financial Analyst System.

Reads batch validation CSV and failure analysis JSON, then builds a
calibration profile Gemini can reference when making predictions.

Profile includes:
- hold_rate, false_buy_rate, false_sell_rate
- avg_return_error
- failure categories
- recommended improvements
- data quality warnings
- global profile, by_horizon breakdown, prompt_guidance

Usage:
    python calibration_profile_builder.py
    python calibration_profile_builder.py --symbol SPY TSLA
    python calibration_profile_builder.py --tag baseline
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

ROOT             = Path(__file__).resolve().parent
BATCH_CSV        = ROOT / "data" / "batch_validation_results.csv"
FAILURE_JSON     = ROOT / "data" / "failure_analysis.json"
PROFILE_JSON     = ROOT / "data" / "calibration_profiles.json"

FOCUSED_SYMBOLS = ["SPY", "TSLA", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "QQQ", "IWM"]


def _direction_match(ai: Optional[str], actual: Optional[str]) -> bool:
    if not ai or not actual:
        return False
    ai_up  = ai.upper()
    act_up = actual.upper()
    pos = {"BUY"}
    neg = {"SELL"}
    if ai_up in pos and act_up in pos:
        return True
    if ai_up in neg and act_up in neg:
        return True
    return ai_up == act_up


def _build_global_profile(rows: List[dict], fail_data: dict) -> dict:
    """Build global (across all symbols) calibration metrics."""
    valid = [r for r in rows if r.get("validation_status") not in ("ERROR", "DATA_UNAVAILABLE", None)]
    n = len(valid)
    if n == 0:
        return {"n": 0, "note": "No valid rows"}

    matches     = [r for r in valid if r.get("agreement") == "MATCH"]
    holds       = [r for r in valid if (r.get("ai_decision") or "").upper() == "HOLD"]
    false_buys  = [r for r in valid
                   if (r.get("ai_decision") or "").upper() == "BUY"
                   and (r.get("actual_decision") or "").upper() != "BUY"]
    false_sells = [r for r in valid
                   if (r.get("ai_decision") or "").upper() == "SELL"
                   and (r.get("actual_decision") or "").upper() != "SELL"]
    over_holds  = [r for r in valid
                   if (r.get("ai_decision") or "").upper() == "HOLD"
                   and (r.get("actual_decision") or "").upper() in ("BUY", "SELL")]

    dir_matches = [r for r in valid if _direction_match(r.get("ai_decision"), r.get("actual_decision"))]
    errs = [abs(float(r["return_error_pp"])) for r in valid
            if r.get("return_error_pp") not in (None, "", "None")]

    # Best/worst horizon by match rate
    h_match: Dict[str, float] = {}
    for h in set(str(r.get("horizon_days", "")) for r in valid if r.get("horizon_days") not in (None, "", "None")):
        h_rows = [r for r in valid if str(r.get("horizon_days", "")) == h]
        h_m    = [r for r in h_rows if r.get("agreement") == "MATCH"]
        if h_rows:
            h_match[h] = round(len(h_m) / len(h_rows) * 100, 1)
    best_h  = max(h_match.items(), key=lambda x: x[1])[0] if h_match else None
    worst_h = min(h_match.items(), key=lambda x: x[1])[0] if h_match else None

    return {
        "n": n,
        "decision_match_pct":   round(len(matches) / n * 100, 1),
        "direction_match_pct":  round(len(dir_matches) / n * 100, 1),
        "avg_return_error_pp":  round(sum(errs) / len(errs), 2) if errs else None,
        "hold_rate":            round(len(holds) / n * 100, 1),
        "buy_rate":             round(sum(1 for r in valid if (r.get("ai_decision") or "").upper() == "BUY") / n * 100, 1),
        "sell_rate":            round(sum(1 for r in valid if (r.get("ai_decision") or "").upper() == "SELL") / n * 100, 1),
        "false_buy_rate":       round(len(false_buys) / n * 100, 1),
        "false_sell_rate":      round(len(false_sells) / n * 100, 1),
        "over_hold_rate":       round(len(over_holds) / n * 100, 1),
        "best_horizon":         best_h,
        "worst_horizon":        worst_h,
    }


def _build_horizon_profile(rows: List[dict]) -> dict:
    """Build per-horizon calibration metrics."""
    valid = [r for r in rows if r.get("validation_status") not in ("ERROR", "DATA_UNAVAILABLE", None)]
    result: Dict[str, dict] = {}
    for h in sorted(set(str(r.get("horizon_days", "")) for r in valid) - {"", "None"}):
        h_rows  = [r for r in valid if str(r.get("horizon_days", "")) == h]
        n       = len(h_rows)
        matches = [r for r in h_rows if r.get("agreement") == "MATCH"]
        holds   = [r for r in h_rows if (r.get("ai_decision") or "").upper() == "HOLD"]
        errs    = [abs(float(r["return_error_pp"])) for r in h_rows
                   if r.get("return_error_pp") not in (None, "", "None")]
        result[h] = {
            "runs":               n,
            "decision_match_pct": round(len(matches) / n * 100, 1) if n else 0,
            "hold_rate":          round(len(holds) / n * 100, 1) if n else 0,
            "avg_return_error_pp":round(sum(errs) / len(errs), 2) if errs else None,
        }
    return result


def _build_prompt_guidance(profiles: Dict[str, dict], global_p: dict, horizon_p: dict) -> List[str]:
    """Build a list of prompt guidance warning strings."""
    guidance = []

    hold_rate = global_p.get("hold_rate", 0) or 0
    if hold_rate > 45:
        guidance.append(
            f"GLOBAL HOLD RATE: {hold_rate}% — OVER_HOLD is the primary accuracy enemy. "
            "Force BUY/SELL when signals agree."
        )

    for sym, p in profiles.items():
        if not p.get("calibration_available"):
            continue
        sym_hold = p.get("hold_rate_pct", 0) or 0
        if sym_hold > 50:
            guidance.append(
                f"SYMBOL WARNING: {sym} has {sym_hold}% hold rate — over-holds severely. "
                "Commit to directional call when trend+momentum agree."
            )

    false_buy_rate = global_p.get("false_buy_rate", 0) or 0
    if false_buy_rate > 15:
        guidance.append(
            f"FALSE BUY WARNING: {false_buy_rate}% of AI BUY calls are wrong. "
            "Add reversal-risk check and RSI overbought guard before BUY."
        )

    false_sell_rate = global_p.get("false_sell_rate", 0) or 0
    if false_sell_rate > 15:
        guidance.append(
            f"FALSE SELL WARNING: {false_sell_rate}% of AI SELL calls are wrong. "
            "Add RSI oversold check and near-support guard before SELL."
        )

    avg_err = global_p.get("avg_return_error_pp")
    if avg_err is not None and avg_err > 10:
        guidance.append(
            f"MAGNITUDE WARNING: Average error is {avg_err}pp — "
            "Gemini over/under-predicts return magnitude. Recalibrate return scaling."
        )

    if not guidance:
        guidance.append("No dominant failure patterns detected. Continue monitoring.")

    return guidance


def build_profiles(
    symbols: Optional[List[str]] = None,
    csv_path: Path = BATCH_CSV,
    fail_json: Path = FAILURE_JSON,
    tag: str = "",
) -> dict:
    """Build calibration profiles for each symbol."""
    symbols = symbols or FOCUSED_SYMBOLS

    rows: List[dict] = []
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    fail_data: dict = {}
    if fail_json.exists():
        fail_data = json.loads(fail_json.read_text(encoding="utf-8"))

    profiles: Dict[str, dict] = {}
    for sym in symbols:
        sym_rows = [r for r in rows if r.get("symbol") == sym
                    and r.get("validation_status") not in ("ERROR", "DATA_UNAVAILABLE", None)]
        if not sym_rows:
            profiles[sym] = {
                "symbol": sym,
                "runs_available": 0,
                "data_quality": "INSUFFICIENT_DATA",
                "calibration_available": False,
            }
            continue

        n = len(sym_rows)
        matches    = [r for r in sym_rows if r.get("agreement") == "MATCH"]
        holds      = [r for r in sym_rows if (r.get("ai_decision") or "").upper() == "HOLD"]
        false_buys = [r for r in sym_rows
                      if (r.get("ai_decision") or "").upper() == "BUY"
                      and (r.get("actual_decision") or "").upper() != "BUY"]
        false_sells = [r for r in sym_rows
                       if (r.get("ai_decision") or "").upper() == "SELL"
                       and (r.get("actual_decision") or "").upper() != "SELL"]

        ret_errors = [
            abs(float(r["return_error_pp"]))
            for r in sym_rows
            if r.get("return_error_pp") not in (None, "", "None")
        ]

        sym_fail = fail_data.get("by_symbol", {}).get(sym, {})
        top_fails = sym_fail.get("top_failures", {})

        data_quality = "SUFFICIENT" if n >= 10 else "LIMITED" if n >= 3 else "INSUFFICIENT_DATA"
        vol_warning  = any("VOLATILITY" in f or "REVERSAL" in f for f in top_fails)
        news_warning = any("NEWS" in f or "EARNINGS" in f for f in top_fails)

        profiles[sym] = {
            "symbol":               sym,
            "runs_available":       n,
            "calibration_available": True,
            "data_quality":         data_quality,
            "match_pct":            round(len(matches) / n * 100, 1),
            "hold_rate_pct":        round(len(holds) / n * 100, 1),
            "false_buy_rate_pct":   round(len(false_buys) / n * 100, 1),
            "false_sell_rate_pct":  round(len(false_sells) / n * 100, 1),
            "avg_return_error_pp":  round(sum(ret_errors) / len(ret_errors), 4) if ret_errors else None,
            "top_failure_categories": list(top_fails.keys())[:5],
            "volatility_warning":   vol_warning,
            "news_earnings_warning": news_warning,
            "calibration_summary":  _build_summary_text(sym, n, len(matches), holds, false_buys,
                                                         false_sells, top_fails, vol_warning, news_warning),
        }

    global_p  = _build_global_profile(rows, fail_data)
    horizon_p = _build_horizon_profile(rows)
    guidance  = _build_prompt_guidance(profiles, global_p, horizon_p)

    return {
        "profiles":           profiles,
        "global":             global_p,
        "by_horizon":         horizon_p,
        "prompt_guidance":    guidance,
        "symbols_profiled":   list(profiles.keys()),
        "total_profiles":     len(profiles),
        "profiles_with_data": sum(1 for p in profiles.values() if p.get("calibration_available")),
        "generated_at":       datetime.utcnow().isoformat(),
        "tag":                tag if tag else "default",
    }


def _build_summary_text(
    sym, n, n_matches, holds, false_buys, false_sells, top_fails, vol_warn, news_warn
) -> str:
    lines = [
        f"{sym}: {n} validation runs. Match rate {round(n_matches/n*100,1) if n else 0}%.",
    ]
    if len(holds) / max(n, 1) > 0.5:
        lines.append("WARNING: AI over-holds on this symbol — consider lowering HOLD threshold.")
    if false_buys:
        lines.append(f"AI makes {len(false_buys)} false BUY calls — be more cautious on upside.")
    if false_sells:
        lines.append(f"AI makes {len(false_sells)} false SELL calls — avoid early bearish calls.")
    if vol_warn:
        lines.append("Volatility shifts frequently missed — include ATR/vol in context.")
    if news_warn:
        lines.append("Earnings/news events cause AI errors — check calendar before prediction.")
    if top_fails:
        lines.append(f"Top failure category: {list(top_fails.keys())[0]}.")
    return " ".join(lines)


def _print_profiles(result: dict) -> None:
    print(f"\n-- Calibration Profiles ({result['profiles_with_data']}/{result['total_profiles']} with data) --")
    for sym, p in sorted(result["profiles"].items()):
        if p.get("calibration_available"):
            print(f"  {sym:8s}  match={p['match_pct']:5.1f}%  hold={p['hold_rate_pct']:4.1f}%  "
                  f"false_buy={p['false_buy_rate_pct']:4.1f}%  runs={p['runs_available']}")
        else:
            print(f"  {sym:8s}  INSUFFICIENT DATA ({p['runs_available']} runs)")

    gp = result.get("global", {})
    if gp.get("n"):
        print(f"\n  Global:  match={gp.get('decision_match_pct')}%  hold={gp.get('hold_rate')}%  "
              f"false_buy={gp.get('false_buy_rate')}%  over_hold={gp.get('over_hold_rate')}%  n={gp['n']}")

    guidance = result.get("prompt_guidance", [])
    if guidance:
        print("\n  Prompt Guidance:")
        for g in guidance:
            print(f"    - {g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibration Profile Builder")
    parser.add_argument("--symbol", nargs="*", default=FOCUSED_SYMBOLS)
    parser.add_argument("--out",    default="")
    parser.add_argument("--tag",    default="",
                        help="Tag for input/output file naming (e.g. 'baseline')")
    args = parser.parse_args()

    if args.tag:
        csv_path  = ROOT / "data" / f"batch_validation_results_{args.tag}.csv"
        fail_path = ROOT / "data" / f"failure_analysis_{args.tag}.json"
        out_path  = ROOT / "data" / f"calibration_profiles_{args.tag}.json"
    else:
        csv_path  = BATCH_CSV
        fail_path = FAILURE_JSON
        out_path  = Path(args.out) if args.out else PROFILE_JSON

    result = build_profiles(symbols=args.symbol, csv_path=csv_path, fail_json=fail_path, tag=args.tag)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    _print_profiles(result)
    print(f"\n  Full JSON: {out_path}")
