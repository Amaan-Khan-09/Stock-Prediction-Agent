"""
Pending Future Predictions Engine

Manages live future predictions that cannot yet be validated because
the target date has not arrived. Records are saved to
pending_future_predictions.jsonl and can be validated later.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from jsonl_store import append_jsonl

ROOT = Path(__file__).resolve().parent
_PENDING_FILE = ROOT / "pending_future_predictions.jsonl"


def save_pending_prediction(spi: dict, ai_result: dict) -> Tuple[bool, str]:
    """
    Save a live future prediction to pending_future_predictions.jsonl.
    Called when target_date is in the future.
    """
    try:
        record = {
            "record_type":        "pending_future_prediction",
            "status":             "PENDING_VALIDATION",
            "saved_at":           datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "symbol":             spi.get("symbol", ""),
            "historical_context_start_date": spi.get("historical_context_start_date", ""),
            "prediction_origin_date":        spi.get("prediction_origin_date", ""),
            "effective_origin_date":         ai_result.get("effective_origin_date", ""),
            "target_date":                   spi.get("target_date", ""),
            "decision_horizon_days":         spi.get("decision_horizon_days", 30),
            "initial_capital":               spi.get("initial_capital", 50000),
            "price_basis":                   spi.get("price_basis", "close"),
            "benchmark":                     spi.get("benchmark", "SPY"),
            "validation_mode":               spi.get("validation_mode", "horizon_days"),
            "ai_decision":                   ai_result.get("decision", ""),
            "origin_price_used":             ai_result.get("origin_price_used"),
            "predicted_target_price":        ai_result.get("predicted_target_price"),
            "predicted_return_pct":          ai_result.get("predicted_return_pct"),
            "predicted_final_capital":       ai_result.get("predicted_final_capital"),
            "predicted_total_pl":            ai_result.get("predicted_total_pl"),
            "confidence_score":              ai_result.get("confidence_score"),
            "risk_score":                    ai_result.get("risk_score"),
            "data_quality_score":            ai_result.get("data_quality_score"),
            "model_version":                 ai_result.get("model_version", ""),
            "validation_input_hash":         ai_result.get("stock_prediction_input_hash", ""),
            "features_used":                 ai_result.get("features_used", {}),
        }
        append_jsonl(_PENDING_FILE, record)
        return True, "Pending prediction saved."
    except Exception as exc:
        return False, f"Failed to save pending prediction: {exc}"


def load_pending_predictions() -> List[Dict]:
    """Load all records from pending_future_predictions.jsonl."""
    if not _PENDING_FILE.exists():
        return []
    records = []
    with open(_PENDING_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def count_pending_predictions() -> int:
    """Return number of records still waiting for validation."""
    records = load_pending_predictions()
    return sum(1 for r in records if r.get("status") == "PENDING_VALIDATION")


def get_pending_summary() -> Dict[str, Any]:
    """Summary of pending predictions for the accuracy section."""
    records = load_pending_predictions()
    pending = [r for r in records if r.get("status") == "PENDING_VALIDATION"]
    return {
        "total_pending": len(pending),
        "symbols":       list({r["symbol"] for r in pending if r.get("symbol")}),
        "earliest_target": min((r["target_date"] for r in pending if r.get("target_date")), default=""),
        "latest_target":   max((r["target_date"] for r in pending if r.get("target_date")), default=""),
    }
