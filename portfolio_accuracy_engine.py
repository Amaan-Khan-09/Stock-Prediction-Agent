"""
Portfolio Strategy Accuracy Engine
Saves/loads portfolio evaluation records separately from single-stock records.
File: portfolio_strategy_evaluation_runs.jsonl  (NOT strategy_prediction_evaluation_runs.jsonl)
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
import sys

from jsonl_store import append_jsonl

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

PORTFOLIO_EVAL_FILE    = ROOT / "portfolio_strategy_evaluation_runs.jsonl"
PORTFOLIO_MODEL_VERSION = "portfolio_strategy_predictor_v1"


def build_portfolio_evaluation_record(
    portfolio_si:    dict,
    portfolio_hash:  str,
    pf_ai:           dict,
    pf_bt:           dict,
    pf_cmp:          dict,
) -> dict:
    """Build evaluation record for one portfolio strategy run."""
    return {
        "timestamp":               datetime.now(timezone.utc).isoformat(),
        "source":                  "portfolio_strategy_backtest_verification",
        "model_version":           PORTFOLIO_MODEL_VERSION,
        "run_id":                  str(uuid.uuid4()),
        "portfolio_strategy_input": portfolio_si,
        "portfolio_strategy_hash":  portfolio_hash,
        "ai_prediction":            pf_ai,
        "backtest_actual":          pf_bt,
        "comparison":               pf_cmp,
        "holding_comparisons":      pf_cmp.get("holding_comparisons", []),
    }


def save_portfolio_evaluation_record(record: dict) -> None:
    """Append one portfolio evaluation record to the portfolio-specific JSONL file."""
    try:
        append_jsonl(PORTFOLIO_EVAL_FILE, record)
    except Exception:
        pass


def load_portfolio_evaluation_records() -> list:
    """Load all portfolio evaluation records. Returns [] on missing file or parse error."""
    records = []
    if not PORTFOLIO_EVAL_FILE.exists():
        return records
    try:
        with open(PORTFOLIO_EVAL_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except Exception:
        pass
    return records


def calculate_portfolio_accuracy_metrics(records: list) -> dict:
    """
    Compute portfolio accuracy metrics from evaluation records.
    Weights:
      Decision Accuracy  20%
      Directional Acc.   15%
      Return Accuracy    25%
      P&L Accuracy       20%
      Capital Accuracy   10%
      Win Rate Accuracy  10%

    Note: portfolio accuracy measures AI portfolio prediction closeness against
    aggregated tastytrade holding backtests — separate from single-stock accuracy.
    """
    if not records:
        return {"status": "NO_RECORDS", "count": 0}

    valid = [
        r for r in records
        if r.get("ai_prediction") and r.get("backtest_actual")
    ]
    if not valid:
        return {"status": "NO_VALID_RECORDS", "count": 0}

    def _sf(v, d=0.0):
        try:
            return float(v) if v is not None else d
        except (TypeError, ValueError):
            return d

    dec_matches, dir_matches = [], []
    ret_errs, pl_errs, cap_errs, wr_errs = [], [], [], []

    for r in valid:
        ai  = r["ai_prediction"]
        bt  = r["backtest_actual"]

        ai_dec = str(ai.get("decision", "")).upper()
        bt_dec = str(bt.get("decision", "")).upper()
        dec_matches.append(1 if ai_dec == bt_dec else 0)

        ai_ret = _sf(ai.get("total_return_pct"))
        bt_ret = _sf(bt.get("total_return_pct"))
        dir_matches.append(1 if (ai_ret > 0) == (bt_ret > 0) else 0)
        ret_errs.append(abs(ai_ret - bt_ret))

        ai_pl = _sf(ai.get("total_pl"))
        bt_pl = _sf(bt.get("total_pl"))
        pl_errs.append(abs(ai_pl - bt_pl))

        ai_cap = _sf(ai.get("final_capital"))
        bt_cap = _sf(bt.get("final_capital"))
        ic     = _sf(bt.get("initial_capital"), 100_000)
        cap_errs.append(abs(ai_cap - bt_cap) / max(abs(ic), 1) * 100)

        wr_errs.append(abs(_sf(ai.get("win_rate")) - _sf(bt.get("win_rate"))))

    n = len(valid)
    dec_acc = round(sum(dec_matches) / n * 100, 1)
    dir_acc = round(sum(dir_matches) / n * 100, 1)

    avg_ret_err = round(sum(ret_errs) / n, 2)
    avg_pl_err  = round(sum(pl_errs)  / n, 2)
    avg_cap_err = round(sum(cap_errs) / n, 2)
    avg_wr_err  = round(sum(wr_errs)  / n, 2)

    ret_acc = round(max(0.0, 100.0 - min(avg_ret_err, 100.0)), 1)
    # P&L accuracy: relative error on pl magnitude
    max_pl  = max(max(abs(e) for e in pl_errs), 1.0)
    pl_acc  = round(max(0.0, 100.0 - min(sum(pl_errs) / n / max_pl * 100, 100.0)), 1)
    cap_acc = round(max(0.0, 100.0 - min(avg_cap_err, 100.0)), 1)
    wr_acc  = round(max(0.0, 100.0 - min(avg_wr_err, 100.0)), 1)

    overall = round(
        dec_acc * 0.20
        + dir_acc * 0.15
        + ret_acc * 0.25
        + pl_acc  * 0.20
        + cap_acc * 0.10
        + wr_acc  * 0.10,
        1,
    )

    return {
        "status":                     "SUCCESS",
        "count":                      n,
        "decision_accuracy_pct":      dec_acc,
        "directional_accuracy_pct":   dir_acc,
        "return_accuracy_pct":        ret_acc,
        "pl_accuracy_pct":            pl_acc,
        "final_capital_accuracy_pct": cap_acc,
        "win_rate_accuracy_pct":      wr_acc,
        "overall_accuracy_pct":       overall,
        "avg_return_error":           avg_ret_err,
        "avg_pl_error":               avg_pl_err,
        "avg_capital_error_pct":      avg_cap_err,
        "avg_win_rate_error":         avg_wr_err,
    }
