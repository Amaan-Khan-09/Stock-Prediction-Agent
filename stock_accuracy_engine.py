"""
Stock Prediction Accuracy Engine
Saves and loads stock prediction validation records.

Record format (JSONL):
{
    "timestamp": "...",
    "source": "stock_prediction_validation",
    "model_version": "stock_predictor_v1_walkforward",
    "stock_prediction_input": {...},
    "stock_prediction_input_hash": "...",
    "ai_prediction": {
        "predicted_target_price": ...,
        "predicted_return_pct": ...,
        "predicted_final_capital": ...,
        "decision": ...
    },
    "actual_validation": {
        "origin_price": ...,
        "target_price": ...,
        "actual_return_pct": ...,
        "actual_final_capital": ...,
        "actual_decision": ...
    },
    "comparison": {
        "decision_match": ...,
        "directional_match": ...,
        "capital_error": ...,
        "return_error_pct": ...,
        "abs_return_error_pct": ...
    }
}

Excluded records:
- Failed AI predictions (status != SUCCESS)
- Failed validations (status != SUCCESS)
- Missing actual prices
- Future target dates
- Records with missing return/capital values
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jsonl_store import append_jsonl

ROOT             = Path(__file__).resolve().parent
STOCK_EVAL_FILE  = ROOT / "stock_prediction_evaluation_runs.jsonl"
STOCK_INVALID_FILE = ROOT / "stock_prediction_evaluation_runs_invalid.jsonl"


def is_valid_stock_record(record: dict) -> bool:
    """
    Return True if a stock prediction validation record is valid and should be saved.
    Rejects: failed AI, failed validation, missing prices, future dates, no comparison.
    """
    if not isinstance(record, dict):
        return False

    # AI prediction must have succeeded
    ai = record.get("ai_prediction") or {}
    if not isinstance(ai, dict):
        return False
    if not ai.get("decision"):   # None or "" only; REVIEW is a valid AI output
        return False
    if ai.get("predicted_return_pct") is None:
        return False
    if ai.get("predicted_final_capital") is None:
        return False

    # Actual validation must have succeeded
    av = record.get("actual_validation") or {}
    if not isinstance(av, dict):
        return False
    if av.get("origin_price") is None or av.get("target_price") is None:
        return False
    if av.get("actual_return_pct") is None:
        return False
    if av.get("actual_final_capital") is None:
        return False
    if av.get("actual_decision") not in ("BUY", "SELL", "HOLD"):
        return False

    # Comparison must exist and have succeeded.
    # Note: _build_record() explicitly copies "status" from the comparison dict.
    cmp = record.get("comparison") or {}
    if not isinstance(cmp, dict):
        return False
    if cmp.get("status") != "SUCCESS":
        return False
    # decision_match can be False (CONFLICT) — that is a valid outcome, not a failure.
    if cmp.get("decision_match") is None:
        return False

    # Target date must be in the past (can't validate future)
    spi = record.get("stock_prediction_input") or {}
    tgt = spi.get("target_date", "")
    if tgt:
        try:
            if datetime.strptime(str(tgt)[:10], "%Y-%m-%d").date() > date.today():
                return False
        except ValueError:
            return False

    # Symbol must be non-empty
    sym = str(spi.get("symbol", "") or "").strip()
    if not sym:
        return False

    return True


def _build_record(
    spi: dict,
    ai_prediction: dict,
    actual_validation: dict,
) -> dict:
    """Build the standard JSONL record from the three components."""
    cmp = actual_validation.get("comparison", {})
    return {
        "timestamp":                 datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "run_id":                    str(uuid.uuid4())[:8].upper(),
        "source":                    "stock_prediction_validation",
        "model_version":             ai_prediction.get("model_version", ""),
        "stock_prediction_input":    spi,
        "stock_prediction_input_hash": ai_prediction.get("stock_prediction_input_hash", ""),
        "ai_prediction": {
            "predicted_target_price":  ai_prediction.get("predicted_target_price"),
            "predicted_return_pct":    ai_prediction.get("predicted_return_pct"),
            "predicted_final_capital": ai_prediction.get("predicted_final_capital"),
            "predicted_total_pl":      ai_prediction.get("predicted_total_pl"),
            "decision":                ai_prediction.get("decision"),
            "confidence_score":        ai_prediction.get("confidence_score"),
            "risk_score":              ai_prediction.get("risk_score"),
        },
        "actual_validation": {
            "origin_price":          actual_validation.get("origin_price"),
            "target_price":          actual_validation.get("target_price"),
            "effective_origin_date": actual_validation.get("effective_origin_price_date"),
            "effective_target_date": actual_validation.get("effective_target_price_date"),
            "actual_return_pct":     actual_validation.get("actual_return_pct"),
            "actual_final_capital":  actual_validation.get("actual_final_capital"),
            "actual_total_pl":       actual_validation.get("actual_total_pl"),
            "actual_decision":       actual_validation.get("actual_decision"),
        },
        "comparison": {
            "status":                cmp.get("status"),        # required by is_valid_stock_record
            "decision_match":        cmp.get("decision_match"),
            "directional_match":     cmp.get("directional_match"),
            "agreement":             cmp.get("agreement"),     # MATCH or CONFLICT — both valid
            "final_decision":        cmp.get("final_decision"),
            "capital_error":         cmp.get("capital_error"),
            "abs_capital_error":     cmp.get("abs_capital_error"),
            "capital_error_pct":     cmp.get("capital_error_pct"),
            "return_error_pct":      cmp.get("return_error_pct"),
            "abs_return_error_pct":  cmp.get("abs_return_error_pct"),
            "price_error":           cmp.get("price_error"),
            "abs_price_error":       cmp.get("abs_price_error"),
        },
    }


def save_stock_prediction_record(
    spi: dict,
    ai_prediction: dict,
    actual_validation: dict,
) -> Tuple[bool, str]:
    """
    Save a validated stock prediction record to JSONL.
    Returns (saved, message).
    Invalid records are written to the invalid file instead.
    """
    record = _build_record(spi, ai_prediction, actual_validation)
    if is_valid_stock_record(record):
        try:
            append_jsonl(STOCK_EVAL_FILE, record)
            return True, "Record saved."
        except Exception as exc:
            return False, f"Save error: {exc}"
    else:
        # Write to invalid file for traceability
        try:
            append_jsonl(STOCK_INVALID_FILE, record)
        except Exception:
            pass
        return False, "Record failed validation — saved to invalid file."


def load_stock_prediction_records(limit: int = 200) -> List[dict]:
    """Load recent valid records from JSONL, newest first."""
    if not STOCK_EVAL_FILE.exists():
        return []
    records = []
    try:
        with open(STOCK_EVAL_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except Exception:
        pass
    return list(reversed(records))[:limit]


def get_accuracy_summary(records: List[dict]) -> dict:
    """Compute aggregate accuracy stats from a list of records."""
    if not records:
        return {"total": 0, "decision_match_rate": None, "direction_match_rate": None,
                "avg_abs_return_error": None, "avg_abs_capital_error": None}

    dm_count  = 0
    dir_count = 0
    ret_errs  = []
    cap_errs  = []
    for r in records:
        cmp = r.get("comparison") or {}
        if cmp.get("decision_match") is True:   dm_count  += 1
        if cmp.get("directional_match") is True: dir_count += 1
        if cmp.get("abs_return_error_pct") is not None:
            ret_errs.append(float(cmp["abs_return_error_pct"]))
        if cmp.get("abs_capital_error") is not None:
            cap_errs.append(float(cmp["abs_capital_error"]))

    n = len(records)
    return {
        "total":                n,
        "decision_match_rate":  round(dm_count  / n * 100, 1),
        "direction_match_rate": round(dir_count / n * 100, 1),
        "avg_abs_return_error": round(sum(ret_errs) / len(ret_errs), 2) if ret_errs else None,
        "avg_abs_capital_error": round(sum(cap_errs) / len(cap_errs), 2) if cap_errs else None,
    }


