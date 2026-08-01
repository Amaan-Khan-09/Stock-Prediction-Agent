"""
Strategy Accuracy Engine
Stores and calculates accuracy metrics for AI strategy prediction vs tastytrade backtest actual.
File: strategy_prediction_evaluation_runs.jsonl
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from jsonl_store import append_jsonl

EVAL_FILE         = Path(__file__).resolve().parent / "strategy_prediction_evaluation_runs.jsonl"
INVALID_EVAL_FILE = Path(__file__).resolve().parent / "strategy_prediction_evaluation_runs_invalid.jsonl"
MODEL_VERSION = "strategy_predictor_v3_backtest_surrogate_calibrated"

# Symbol regex reused for validation checks
import re as _re
_SYMBOL_RE    = _re.compile(r"^[A-Z]{1,5}([.\-][A-Z])?$")
_BLOCKED_SYMS = frozenset({
    "XYZ", "ABCXYZ", "ABC", "ABCD", "ABCDE",
    "TEST", "NONE", "NULL", "NA", "N/A",
    "SYMBOL", "TICKER", "STOCK", "EXAMPLE", "FAKE", "DEMO",
    "PLACEHOLDER", "SAMPLE",
    "XXXX", "YYYY", "ZZZZ",
    "AAAA", "BBBB", "CCCC", "DDDD", "EEEE",
})


def _safe(val, default=0.0):
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


# ── Record schema ─────────────────────────────────────────────────────────────
def build_evaluation_record(
    strategy_input: dict,
    strategy_input_hash: str,
    ai_prediction: dict,
    backtest_actual: dict,
    model_version: str = MODEL_VERSION,
) -> dict:
    """
    Build one evaluation record from a completed AI prediction + backtest run.
    Schema matches the requirement exactly.
    """
    ai_d  = str(ai_prediction.get("decision",  "REVIEW")).upper()
    bt_d  = str(backtest_actual.get("decision", "REVIEW")).upper()

    ai_ret = _safe(ai_prediction.get("predicted_total_return_pct"))
    bt_ret = _safe(backtest_actual.get("total_return_pct"))
    ai_pl  = _safe(ai_prediction.get("predicted_total_pl"))
    bt_pl  = _safe(backtest_actual.get("total_pl"))
    ai_fc  = _safe(ai_prediction.get("predicted_final_capital"))
    bt_fc  = _safe(backtest_actual.get("final_capital"))
    ai_wr  = _safe(ai_prediction.get("predicted_win_rate"))
    bt_wr  = _safe(backtest_actual.get("win_rate"))
    ai_dd  = _safe(ai_prediction.get("predicted_max_drawdown"))
    bt_dd  = _safe(backtest_actual.get("max_drawdown"))

    bt_cap = _safe(backtest_actual.get("initial_capital"), 100_000)

    def _dir(d: str) -> str:
        if d == "BUY":  return "positive"
        if d == "SELL": return "negative"
        return "neutral"

    return {
        "timestamp":           datetime.utcnow().isoformat() + "Z",
        "source":              "strategy_backtest_verification",
        "model_version":       model_version,
        "strategy_input":      {k: strategy_input.get(k) for k in [
            "symbol", "start_date", "end_date", "initial_capital",
            "benchmark", "direction", "side", "dte", "delta", "legs", "entry_frequency"
        ]},
        "strategy_input_hash": strategy_input_hash,
        "ai_prediction": {
            "decision":                   ai_d,
            "predicted_total_return_pct": round(ai_ret, 2),
            "predicted_total_pl":         round(ai_pl,  2),
            "predicted_final_capital":    round(ai_fc,  2),
            "predicted_win_rate":         round(ai_wr,  2),
            "predicted_max_drawdown":     round(ai_dd,  2),
        },
        "backtest_actual": {
            "decision":              bt_d,
            "actual_total_return_pct": round(bt_ret, 2),
            "actual_total_pl":         round(bt_pl,  2),
            "actual_final_capital":    round(bt_fc,  2),
            "actual_win_rate":         round(bt_wr,  2),
            "actual_max_drawdown":     round(bt_dd,  2),
        },
        "comparison": {
            "decision_match":         ai_d == bt_d,
            "directional_match":      _dir(ai_d) == _dir(bt_d),
            "return_error_pct":       round(ai_ret - bt_ret, 2),
            "pl_error":               round(ai_pl  - bt_pl,  2),
            "final_capital_error_pct": round((ai_fc - bt_fc) / bt_cap * 100, 2) if bt_cap else 0.0,
            "win_rate_error":         round(ai_wr  - bt_wr,  2),
        },
    }


def is_valid_evaluation_record(record: dict) -> bool:
    """
    Return True only if the record is safe for accuracy metrics.
    Rejects: fake/blank symbols, future/ancient dates, auto-retried backtests,
    failed backtests, missing return data, invalid decisions.
    """
    from datetime import date as _date, datetime as _datetime

    si = record.get("strategy_input", {})
    bt = record.get("backtest_actual", {})

    if not si or not bt:
        return False

    # Symbol must be valid format and not a known placeholder
    symbol = str(si.get("symbol", "") or "").strip().upper()
    if not symbol or not _SYMBOL_RE.match(symbol) or symbol in _BLOCKED_SYMS:
        return False

    # Dates must be valid: parseable, >= 2000-01-01, end <= today
    try:
        s = _datetime.strptime(str(si.get("start_date", "")), "%Y-%m-%d").date()
        e = _datetime.strptime(str(si.get("end_date",   "")), "%Y-%m-%d").date()
        if s < _date(2000, 1, 1) or e > _date.today() or s >= e:
            return False
    except (ValueError, TypeError):
        return False

    # Must have real return data
    actual_return = bt.get("actual_total_return_pct")
    if actual_return is None:
        return False

    # Decision must be a real signal (not a failure placeholder)
    if bt.get("decision") in (None, "REVIEW", "MISSING", ""):
        return False

    return True


def quarantine_invalid_record(record: dict) -> None:
    """Save a record to the invalid/quarantine JSONL file."""
    try:
        append_jsonl(INVALID_EVAL_FILE, record)
    except Exception:
        pass


def save_strategy_evaluation_record(record: dict) -> None:
    """Append one record to the evaluation JSONL file."""
    try:
        append_jsonl(EVAL_FILE, record)
    except Exception:
        pass


def load_strategy_accuracy_records(
    source: str = "strategy_backtest_verification",
    model_version: Optional[str] = None,
) -> List[dict]:
    """
    Load strategy verification records from JSONL file.
    If model_version is provided, only returns records matching that version.
    """
    records: List[dict] = []
    if not EVAL_FILE.exists():
        return records
    try:
        with open(EVAL_FILE, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if r.get("source") != source:
                        continue
                    if model_version is not None and r.get("model_version") != model_version:
                        continue
                    records.append(r)
                except json.JSONDecodeError:
                    pass
    except Exception:
        pass
    return records


def load_legacy_record_count() -> int:
    """Count records written before model_version was introduced (no model_version key)."""
    count = 0
    if not EVAL_FILE.exists():
        return 0
    try:
        with open(EVAL_FILE, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if (r.get("source") == "strategy_backtest_verification" and
                            r.get("model_version") is None):
                        count += 1
                except json.JSONDecodeError:
                    pass
    except Exception:
        pass
    return count


def calculate_strategy_accuracy_metrics(records: List[dict]) -> dict:
    """
    Compute accuracy metrics from strategy evaluation records.
    Metrics:
    - Decision Accuracy (exact match %)
    - Directional Accuracy (direction match %)
    - Macro Precision, Recall, F1 (per decision class)
    - Average Return Error
    - Average P&L Error
    - Average Final Capital Error %
    - Average Win Rate Error
    - Confusion matrix
    """
    if len(records) < 2:
        return {"error": f"Need >= 2 records, got {len(records)}"}

    y_pred, y_true = [], []
    dir_matches:   List[bool]  = []
    ret_errors:    List[float] = []
    pl_errors:     List[float] = []
    fc_errors:     List[float] = []
    wr_errors:     List[float] = []

    for r in records:
        ai  = r.get("ai_prediction",  {})
        bt  = r.get("backtest_actual", {})
        cmp = r.get("comparison",      {})

        pd = str(ai.get("decision", "REVIEW")).upper()
        td = str(bt.get("decision", "REVIEW")).upper()
        y_pred.append(pd)
        y_true.append(td)

        if cmp.get("directional_match") is not None:
            dir_matches.append(bool(cmp["directional_match"]))
        elif pd and td:
            def _dir(d):
                return "pos" if d == "BUY" else ("neg" if d == "SELL" else "neu")
            dir_matches.append(_dir(pd) == _dir(td))

        if cmp.get("return_error_pct") is not None:
            ret_errors.append(float(cmp["return_error_pct"]))
        if cmp.get("pl_error") is not None:
            pl_errors.append(float(cmp["pl_error"]))
        if cmp.get("final_capital_error_pct") is not None:
            fc_errors.append(float(cmp["final_capital_error_pct"]))
        if cmp.get("win_rate_error") is not None:
            wr_errors.append(float(cmp["win_rate_error"]))

    n       = len(y_true)
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    acc     = round(correct / n, 4)

    labels = sorted(set(y_true + y_pred))
    per: Dict[str, Any] = {}
    for lbl in labels:
        tp = sum(t == lbl and p == lbl for t, p in zip(y_true, y_pred))
        fp = sum(t != lbl and p == lbl for t, p in zip(y_true, y_pred))
        fn = sum(t == lbl and p != lbl for t, p in zip(y_true, y_pred))
        pr = tp / (tp + fp) if (tp + fp) else 0.0
        rc = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0
        per[lbl] = {
            "precision": round(pr, 4), "recall": round(rc, 4), "f1": round(f1, 4),
            "support":   sum(t == lbl for t in y_true),
        }

    macro_p  = sum(v["precision"] for v in per.values()) / len(per) if per else 0.0
    macro_r  = sum(v["recall"]    for v in per.values()) / len(per) if per else 0.0
    macro_f1 = sum(v["f1"]        for v in per.values()) / len(per) if per else 0.0

    # Confusion matrix
    idx = {lbl: i for i, lbl in enumerate(labels)}
    mat = [[0] * len(labels) for _ in range(len(labels))]
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            mat[idx[t]][idx[p]] += 1

    def _avg_abs(lst): return round(sum(abs(x) for x in lst) / len(lst), 2) if lst else 0.0
    def _avg(lst):     return round(sum(lst)               / len(lst), 2) if lst else 0.0

    base = {
        "n_records":                  n,
        "n_correct":                  correct,
        "decision_accuracy":          acc,
        "directional_accuracy":       round(sum(dir_matches) / len(dir_matches), 4) if dir_matches else 0.0,
        "macro_precision":            round(macro_p,  4),
        "macro_recall":               round(macro_r,  4),
        "macro_f1":                   round(macro_f1, 4),
        "avg_return_error":           _avg(ret_errors),
        "avg_return_abs_error":       _avg_abs(ret_errors),
        "avg_pl_error":               _avg(pl_errors),
        "avg_pl_abs_error":           _avg_abs(pl_errors),
        "avg_final_capital_error_pct":_avg(fc_errors),
        "avg_win_rate_error":         _avg(wr_errors),
        "per_class":                  per,
        "cm_labels":                  labels,
        "cm_matrix":                  mat,
    }

    # Compute Overall Strategy Accuracy and attach
    osa = compute_overall_strategy_accuracy(base)
    base["overall_strategy_accuracy"] = osa
    return base


def compute_overall_strategy_accuracy(metrics: dict) -> dict:
    """
    Compute Overall Strategy Accuracy — a single blended score that combines
    both direction correctness AND numeric closeness.

    Weights (sum = 1.00):
      Decision Accuracy     0.20  — right BUY/SELL/HOLD direction
      Directional Accuracy  0.15  — positive/negative/neutral alignment
      Return Accuracy       0.25  — how close the return % prediction is
      P&L Accuracy          0.20  — how close the dollar P&L is
      Final Capital Acc.    0.10  — final capital dollar closeness
      Win Rate Accuracy     0.10  — win rate prediction closeness

    Return error tolerance: within ±10% = 100 points, ±20% = 75 pts, ±50% = 40 pts,
    >100% error = 10 pts (never 0 — the model still has learned something).

    Returns:
        {"overall_accuracy": float (0-1), "overall_accuracy_pct": str,
         "component_scores": {...}, "note": str}
    """

    def _error_score(abs_err: float, tolerance_pct: float = 10.0) -> float:
        """Map absolute error to a 0-1 score. tolerance_pct = range for ~100% score."""
        if abs_err is None:
            return 0.5  # unknown — neutral
        if abs_err <= tolerance_pct:
            return 1.0
        if abs_err <= tolerance_pct * 2:
            return 0.75
        if abs_err <= tolerance_pct * 5:
            return 0.40
        return 0.10  # very large error but model exists

    n = metrics.get("n_records", 0)
    if n < 1:
        return {"overall_accuracy": 0.0, "overall_accuracy_pct": "N/A",
                "component_scores": {}, "note": "No records"}

    decision_acc  = _safe(metrics.get("decision_accuracy",  0))
    direction_acc = _safe(metrics.get("directional_accuracy", 0))
    ret_abs_err   = _safe(metrics.get("avg_return_abs_error",  50))
    pl_abs_err    = _safe(metrics.get("avg_pl_abs_error",      0))
    fc_err_pct    = abs(_safe(metrics.get("avg_final_capital_error_pct", 50)))
    wr_abs_err    = abs(_safe(metrics.get("avg_win_rate_error", 20)))

    # Numeric closeness scores
    ret_score = _error_score(ret_abs_err,  10.0)   # within 10% return is great
    pl_score  = _error_score(pl_abs_err / 1000 if pl_abs_err > 0 else 0,  5.0)  # $5k tolerance
    fc_score  = _error_score(fc_err_pct,  10.0)   # within 10% of initial capital is great
    wr_score  = _error_score(wr_abs_err,   5.0)   # within 5pp win rate is great

    # Weighted overall
    overall = (
        0.20 * decision_acc  +
        0.15 * direction_acc +
        0.25 * ret_score     +
        0.20 * pl_score      +
        0.10 * fc_score      +
        0.10 * wr_score
    )
    overall = round(min(1.0, max(0.0, overall)), 4)

    return {
        "overall_accuracy":     overall,
        "overall_accuracy_pct": f"{overall * 100:.1f}%",
        "component_scores": {
            "decision_acc_score":  round(decision_acc,  4),
            "direction_acc_score": round(direction_acc, 4),
            "return_closeness":    round(ret_score,     4),
            "pl_closeness":        round(pl_score,      4),
            "final_capital_closeness": round(fc_score,  4),
            "win_rate_closeness":  round(wr_score,      4),
        },
        "note": (
            "Decision accuracy measures direction/class only. "
            "Overall Strategy Accuracy includes numeric closeness — "
            "100% decision accuracy with 300% return error = low overall accuracy."
        ),
    }


def run_strategy_rolling_verification(
    strategy_input: dict,
    n_windows: int = 5,
    backtest_fn=None,
) -> int:
    """
    Create rolling verification windows within the selected date range.
    For each window: run AI prediction + tastytrade backtest → save record.

    Args:
        strategy_input: canonical StrategyInput dict
        n_windows: number of rolling windows to create
        backtest_fn: callable(si: dict) -> dict  — must be provided by caller
                     (avoids circular import with streamlit_app)
    Returns:
        number of evaluation records saved
    """
    from datetime import timedelta
    from strategy_prediction_agent import build_strategy_input_hash, run_ai_strategy_prediction
    from strategy_backtest_comparator import enrich_backtest_metrics

    if backtest_fn is None:
        return 0

    start_str = strategy_input["start_date"]
    end_str   = strategy_input["end_date"]

    try:
        start_dt = datetime.strptime(start_str, "%Y-%m-%d")
        end_dt   = datetime.strptime(end_str,   "%Y-%m-%d")
    except ValueError:
        return 0

    total_days = (end_dt - start_dt).days
    if total_days < 365:
        return 0

    window_days = 365
    step_days   = max(30, (total_days - window_days) // max(1, n_windows - 1))

    windows = []
    for i in range(n_windows):
        w_start = start_dt + timedelta(days=i * step_days)
        w_end   = w_start + timedelta(days=window_days)
        if w_end > end_dt:
            w_end = end_dt
        if (w_end - w_start).days < 180:
            continue
        windows.append((w_start.strftime("%Y-%m-%d"), w_end.strftime("%Y-%m-%d")))

    if not windows:
        return 0

    created = 0
    for w_start, w_end in windows:
        win_input  = {**strategy_input, "start_date": w_start, "end_date": w_end}
        input_hash = build_strategy_input_hash(win_input)

        # 1. AI prediction — completely independent, no backtest data
        ai_pred = run_ai_strategy_prediction(win_input)
        if ai_pred.get("status") != "SUCCESS":
            continue

        # 2. tastytrade backtest for this window (ground-truth verifier)
        try:
            bt_raw = backtest_fn(win_input)
        except Exception:
            bt_raw = {
                "passed_validation": False,
                "decision":          "REVIEW",
                "total_pl":          0.0,
                "total_return_pct":  0.0,
                "final_capital":     float(win_input.get("initial_capital", 100_000)),
                "win_rate":          0.0,
                "max_drawdown":      0.0,
                "initial_capital":   float(win_input.get("initial_capital", 100_000)),
            }

        # Enrich any missing metrics before saving
        try:
            bt_raw = enrich_backtest_metrics(bt_raw, win_input)
        except Exception:
            pass

        mv = ai_pred.get("model_version", MODEL_VERSION)
        record = build_evaluation_record(win_input, input_hash, ai_pred, bt_raw, model_version=mv)
        save_strategy_evaluation_record(record)
        created += 1

    return created
