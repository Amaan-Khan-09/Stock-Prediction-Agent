"""
compare_calibration_runs.py — Before/After Calibration Comparison

Usage:
    python compare_calibration_runs.py --baseline baseline --post post_calibration
    python compare_calibration_runs.py  (compares default vs latest)

Output:
    data/calibration_before_after_report.json
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def load_summary(tag: str) -> dict:
    fname = f"batch_validation_summary_{tag}.json" if tag else "batch_validation_summary.json"
    path = DATA / fname
    if not path.exists():
        return {"error": f"File not found: {path}"}
    return json.loads(path.read_text(encoding="utf-8"))


def compare(
    baseline_tag: str = "baseline",
    post_tag: str = "post_calibration",
    output_path: Optional[Path] = None,
) -> dict:
    """Compare two tagged batch_validation_summary files.

    output_path defaults to the real data/calibration_before_after_report.json
    for normal CLI use. Tests that pass canned/mocked summaries MUST pass a
    tmp_path-based output_path here instead -- previously this always wrote to
    the real project data/ folder regardless of where load_summary's data came
    from, so running the test suite silently overwrote the real calibration
    report with fixture numbers that were never a genuine comparison run.
    """
    base = load_summary(baseline_tag)
    post = load_summary(post_tag)

    if base.get("error"):
        return {"error": f"Baseline missing: {base['error']}"}
    if post.get("error"):
        return {"error": f"Post-calibration missing: {post['error']}"}

    def delta(key):
        b = base.get(key)
        p = post.get(key)
        if b is None or p is None:
            return None
        return round(p - b, 2)

    dm_delta  = delta("decision_match_pct")
    dir_delta = delta("direction_match_pct")
    hold_delta = delta("hold_rate_pct")
    err_delta  = delta("avg_return_error_pp")

    improved  = []
    regressed = []

    if dm_delta is not None:
        if dm_delta > 0:
            improved.append(f"Decision match improved +{dm_delta}pp")
        elif dm_delta < 0:
            regressed.append(f"Decision match worsened {dm_delta}pp")

    if hold_delta is not None:
        if hold_delta < 0:
            improved.append(f"Hold rate reduced {hold_delta}pp")
        elif hold_delta > 2:
            regressed.append(f"Hold rate increased +{hold_delta}pp")

    if err_delta is not None:
        if err_delta < 0:
            improved.append(f"Return error reduced {err_delta}pp")
        elif err_delta > 0:
            regressed.append(f"Return error worsened +{err_delta}pp")

    honest_assessment = "NO IMPROVEMENT YET — further feature/data work required."
    if improved and not regressed:
        honest_assessment = f"IMPROVEMENT: {'; '.join(improved)}"
    elif improved and regressed:
        honest_assessment = f"MIXED: improved in {len(improved)} metric(s), regressed in {len(regressed)}."

    report = {
        "generated_at":             datetime.utcnow().isoformat(),
        "baseline_tag":             baseline_tag,
        "post_tag":                 post_tag,
        "baseline_decision_match":  base.get("decision_match_pct"),
        "post_decision_match":      post.get("decision_match_pct"),
        "decision_match_delta":     dm_delta,
        "baseline_direction_match": base.get("direction_match_pct"),
        "post_direction_match":     post.get("direction_match_pct"),
        "direction_match_delta":    dir_delta,
        "baseline_hold_rate":       base.get("hold_rate_pct"),
        "post_hold_rate":           post.get("hold_rate_pct"),
        "hold_rate_delta":          hold_delta,
        "baseline_avg_error":       base.get("avg_return_error_pp"),
        "post_avg_error":           post.get("avg_return_error_pp"),
        "avg_error_delta":          err_delta,
        "baseline_valid_runs":      base.get("valid_runs"),
        "post_valid_runs":          post.get("valid_runs"),
        "improved":                 improved,
        "regressed":                regressed,
        "honest_assessment":        honest_assessment,
        "limitations": [
            "Batch runs use historical data from same provider — may share data quality issues.",
            "Sample size may be insufficient for statistical significance.",
            "Results are for backtesting research only — not production trading accuracy.",
        ],
    }

    out = output_path or (DATA / "calibration_before_after_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _print_report(r: dict) -> None:
    print(f"\n-- Before/After Calibration Comparison --")
    if r.get("error"):
        print(f"  ERROR: {r['error']}")
        return
    print(f"  Baseline:  {r['baseline_tag']}")
    print(f"  Post:      {r['post_tag']}")

    dm_d   = r["decision_match_delta"]
    dir_d  = r["direction_match_delta"]
    hold_d = r["hold_rate_delta"]
    err_d  = r["avg_error_delta"]

    dm_str   = f"  Decision Match:    {r['baseline_decision_match']}% -> {r['post_decision_match']}%  (d{dm_d:+}pp)" if dm_d is not None else "  Decision Match:   N/A"
    dir_str  = f"  Direction Match:   {r['baseline_direction_match']}% -> {r['post_direction_match']}%  (d{dir_d:+}pp)" if dir_d is not None else "  Direction Match:  N/A"
    hold_str = f"  Hold Rate:         {r['baseline_hold_rate']}% -> {r['post_hold_rate']}%  (d{hold_d:+}pp)" if hold_d is not None else "  Hold Rate:        N/A"
    err_str  = f"  Avg Return Error:  {r['baseline_avg_error']}pp -> {r['post_avg_error']}pp  (d{err_d:+}pp)" if err_d is not None else "  Avg Return Error: N/A"

    print(dm_str)
    print(dir_str)
    print(hold_str)
    print(err_str)
    print(f"\n  {r['honest_assessment']}")
    for lim in r.get("limitations", []):
        print(f"  ! {lim}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--post",     default="post_calibration")
    parser.add_argument("--out",      default=str(DATA / "calibration_before_after_report.json"))
    args = parser.parse_args()

    result = compare(args.baseline, args.post, output_path=Path(args.out))
    _print_report(result)
    if not result.get("error"):
        print(f"\n  Full JSON: {args.out}")
