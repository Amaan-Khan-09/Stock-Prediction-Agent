"""
clean_accuracy_records.py -- Accuracy Record Hygiene

Reads stock_prediction_evaluation_runs.jsonl and classifies each record:
  - VALID:         accuracy_eligible=true + status=validated + all required fields
  - INVALID:       missing key fields, wrong status, no input_hash
  - STALE:         validation_status=pending + target_date has passed
  - PROXY:         validation_type=DELTA_PROXY_APPROXIMATE or exact_strike_unsupported
  - FAILED_RUN:    accuracy_saved=false or status=FAILED/ERROR/BLOCKED
  - MISSING_HASH:  input_hash absent

Modes:
  --dry-run (default): Report what would change. No file modification.
  --apply:             Write cleaned version to a new file + backup original.

Usage:
    python clean_accuracy_records.py --dry-run
    python clean_accuracy_records.py --apply

Output:
    data/accuracy_cleaned.jsonl  (apply mode only)
    data/accuracy_backup_<ts>.jsonl (backup, apply mode only)
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path

ROOT      = Path(__file__).resolve().parent
# Support eval file at root level (primary) or under data/ (legacy)
_ROOT_FILE = ROOT / "stock_prediction_evaluation_runs.jsonl"
DATA_DIR   = ROOT / "data"
EVAL_FILE  = _ROOT_FILE if _ROOT_FILE.exists() else DATA_DIR / "stock_prediction_evaluation_runs.jsonl"
OUT_FILE   = DATA_DIR / "accuracy_cleaned.jsonl"

TODAY = date.today()


def _get(rec: dict, *keys: str):
    """Get nested value; checks each key at top-level, then falls back to nested lookups."""
    for k in keys:
        if rec.get(k) is not None:
            return rec[k]
    # Try dotted path lookups (e.g. "comparison.status")
    for k in keys:
        if "." in k:
            obj, field = k.split(".", 1)
            nested = rec.get(obj)
            if isinstance(nested, dict) and nested.get(field) is not None:
                return nested[field]
    return None


def _input_hash(rec: dict) -> str | None:
    """Return input hash regardless of field naming variant."""
    return (
        rec.get("input_hash")
        or rec.get("stock_prediction_input_hash")
        or rec.get("options_input_hash")
        or rec.get("run_id")  # fallback: use run_id as unique key
    )


def _rec_status(rec: dict) -> str:
    """Return the top-level status string uppercased."""
    val = (
        rec.get("validation_status")
        or rec.get("status")
        or _get(rec, "comparison.status")
        or ""
    )
    return val.upper()


def _actual_decision(rec: dict) -> str | None:
    return (
        rec.get("actual_decision")
        or _get(rec, "actual_validation.actual_decision")
    )


def _actual_return(rec: dict):
    return (
        rec.get("actual_return_pct")
        or _get(rec, "actual_validation.actual_return_pct")
    )


def _ai_decision(rec: dict) -> str:
    return (
        rec.get("ai_decision")
        or _get(rec, "ai_prediction.decision")
        or ""
    ).upper()


def _agreement(rec: dict) -> str:
    return (
        rec.get("agreement")
        or _get(rec, "comparison.agreement")
        or ""
    ).upper()


def _symbol(rec: dict) -> str | None:
    return (
        rec.get("symbol")
        or _get(rec, "stock_prediction_input.symbol")
        or _get(rec, "options_input.symbol")
    )


def _target_date(rec: dict) -> str | None:
    return (
        rec.get("target_date")
        or _get(rec, "stock_prediction_input.target_date")
        or _get(rec, "options_input.target_date")
    )


def _origin_date(rec: dict) -> str | None:
    return (
        rec.get("prediction_origin_date")
        or _get(rec, "stock_prediction_input.prediction_origin_date")
        or _get(rec, "options_input.prediction_origin_date")
    )


def _classify(rec: dict) -> str:
    """Return the classification for a single record."""
    # Missing hash = cannot trust as unique run
    if not _input_hash(rec):
        return "MISSING_HASH"

    # Proxy / exact-strike runs are not exact-accuracy
    vtype        = (rec.get("validation_type") or "").upper()
    exact_status = (rec.get("exact_validation_status") or "").upper()
    if "PROXY" in vtype or "UNSUPPORTED" in exact_status:
        return "PROXY"

    # Explicitly failed or blocked
    acc_saved  = rec.get("accuracy_saved", True)
    status     = _rec_status(rec)
    if acc_saved is False or status in (
        "FAILED", "ERROR", "BLOCKED", "BACKTEST_CREATE_FAILED",
        "BACKTEST_POLL_FAILED", "FLAT_NO_TRADES", "SKIPPED",
        "DATA_UNAVAILABLE", "REVIEW_REQUIRED", "UNSUPPORTED_BY_PROVIDER",
    ):
        return "FAILED_RUN"

    # Stale — pending with past target date
    if status in ("PENDING", "LIVE_FUTURE_PREDICTION", ""):
        try:
            tgt_str = _target_date(rec) or ""
            tgt = date.fromisoformat(tgt_str)
            if tgt < TODAY:
                return "STALE"
        except (ValueError, TypeError):
            return "STALE"
        return "PENDING"

    # Missing critical validation fields
    if not _symbol(rec) or not _input_hash(rec) or not _target_date(rec) or not _origin_date(rec):
        return "INVALID"

    # Must have actual decision or return to count in accuracy
    if not _actual_decision(rec) and _actual_return(rec) is None:
        return "INVALID"

    return "VALID"


def analyze(path: Path = EVAL_FILE) -> dict:
    if not path.exists():
        return {"error": f"File not found: {path}", "total": 0}

    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                records.append({"_parse_error": f"line {i}", "_raw": line[:80]})

    counts: dict[str, list] = defaultdict(list)
    for rec in records:
        cls = classify_record(rec)
        counts[cls].append(rec)

    valid_recs = counts.get("VALID", [])
    n_valid    = len(valid_recs)

    # Decision match % among VALID only
    matches  = [r for r in valid_recs if _agreement(r) == "MATCH"]
    match_pct = round(len(matches) / n_valid * 100, 1) if n_valid else 0.0

    # Hold rate among VALID
    holds    = [r for r in valid_recs if _ai_decision(r) == "HOLD"]
    hold_pct = round(len(holds) / n_valid * 100, 1) if n_valid else 0.0

    return {
        "file":                  str(path),
        "total_records":         len(records),
        "valid_count":           len(counts.get("VALID", [])),
        "invalid_count":         len(counts.get("INVALID", [])),
        "missing_hash_count":    len(counts.get("MISSING_HASH", [])),
        "stale_count":           len(counts.get("STALE", [])),
        "proxy_count":           len(counts.get("PROXY", [])),
        "failed_run_count":      len(counts.get("FAILED_RUN", [])),
        "pending_count":         len(counts.get("PENDING", [])),
        "decision_match_pct":    match_pct,
        "hold_rate_pct":         hold_pct,
        "to_quarantine":         (
            len(counts.get("INVALID", [])) +
            len(counts.get("MISSING_HASH", [])) +
            len(counts.get("STALE", [])) +
            len(counts.get("PROXY", [])) +
            len(counts.get("FAILED_RUN", []))
        ),
        "by_category":           {k: len(v) for k, v in counts.items()},
        "records_by_category":   counts,
    }


def classify_record(rec: dict) -> str:
    if "_parse_error" in rec:
        return "INVALID"
    return _classify(rec)


def apply_cleanup(path: Path = EVAL_FILE, out: Path = OUT_FILE) -> dict:
    """Write cleaned file (VALID + PENDING only). Back up original first."""
    result = analyze(path)
    if result.get("error"):
        return result

    # Backup
    ts      = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    bak_dir = DATA_DIR
    bak_dir.mkdir(parents=True, exist_ok=True)
    bak     = bak_dir / f"accuracy_backup_{ts}.jsonl"
    shutil.copy2(path, bak)
    result["backup_file"] = str(bak)

    recs = result["records_by_category"]
    keep = recs.get("VALID", []) + recs.get("PENDING", [])

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for rec in keep:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    result["output_file"]   = str(out)
    result["records_kept"]  = len(keep)
    result["records_removed"] = result["total_records"] - len(keep)
    return result


def _print_report(result: dict) -> None:
    if result.get("error"):
        print(f"\n  ERROR: {result['error']}")
        return
    total = result["total_records"]
    print(f"\n-- Accuracy Record Hygiene Report --\n")
    print(f"  File:                  {result['file']}")
    print(f"  Total records:         {total}")
    print(f"  VALID (headline):      {result['valid_count']}")
    print(f"  PENDING (future):      {result.get('pending_count', 0)}")
    print(f"  STALE (past + pending):{result['stale_count']}")
    print(f"  PROXY (delta approx):  {result['proxy_count']}")
    print(f"  FAILED_RUN:            {result['failed_run_count']}")
    print(f"  INVALID (bad fields):  {result['invalid_count']}")
    print(f"  MISSING_HASH:          {result['missing_hash_count']}")
    print(f"  To quarantine:         {result['to_quarantine']}")
    print(f"\n  Valid-only accuracy:")
    print(f"    Decision Match %:    {result['decision_match_pct']}%")
    print(f"    Hold Rate %:         {result['hold_rate_pct']}%")
    if result.get("backup_file"):
        print(f"\n  Backup:              {result['backup_file']}")
    if result.get("output_file"):
        print(f"  Cleaned output:      {result['output_file']}")
        print(f"  Records kept:        {result['records_kept']}")
        print(f"  Records removed:     {result['records_removed']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Accuracy Record Hygiene")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Report only, no changes (default)")
    parser.add_argument("--apply",   action="store_true", default=False,
                        help="Write cleaned file + backup")
    parser.add_argument("--input",   default=str(EVAL_FILE))
    parser.add_argument("--out",     default=str(OUT_FILE))
    args = parser.parse_args()

    if args.apply:
        print("\nApplying cleanup (backup will be created)...")
        result = apply_cleanup(Path(args.input), Path(args.out))
    else:
        print("\nDry-run mode (no changes will be made)...")
        result = analyze(Path(args.input))

    _print_report(result)
    print()
