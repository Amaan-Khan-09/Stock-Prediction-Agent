"""
Recommendation accuracy engine.

This module does not invent labels. It only evaluates agent recommendations
against external recommendation records supplied by a trusted provider export
or integration.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from jsonl_store import append_jsonl


DEFAULT_GROUND_TRUTH_PATHS = [
    os.path.join(os.path.dirname(__file__), "data", "recommendation_ground_truth.csv"),
    os.path.join(os.path.dirname(__file__), "data", "recommendation_ground_truth.jsonl"),
]

RECOMMENDATION_ACCURACY_PATH = os.path.join(os.path.dirname(__file__), "recommendation_accuracy_runs.jsonl")


SOURCE_RESEARCH = [
    {
        "source": "TradingView",
        "status": "LIMITED",
        "reason": "TradingView does not provide an official public historical recommendation API suitable for direct ground-truth backtesting.",
        "use_case": "May be usable only through licensed widgets, broker integrations, exports, or third-party providers.",
    },
    {
        "source": "Financial Modeling Prep",
        "status": "CANDIDATE",
        "reason": "Provides upgrades/downgrades and consensus-style analyst recommendation endpoints; requires API key and plan validation.",
        "use_case": "Best practical candidate for analyst-grade external BUY/HOLD/SELL comparison.",
    },
    {
        "source": "MarketBeat",
        "status": "CANDIDATE",
        "reason": "Provides historical analyst ratings/screener/export capabilities through paid access.",
        "use_case": "Useful if exported to CSV and loaded into data/recommendation_ground_truth.csv.",
    },
    {
        "source": "Tradefeeds",
        "status": "CANDIDATE",
        "reason": "Markets historical/current analyst recommendations, consensus estimates, and price targets via API.",
        "use_case": "Useful paid source for direct historical recommendation snapshots.",
    },
    {
        "source": "Alpha Vantage",
        "status": "PARTIAL",
        "reason": "Strong market/fundamental/news API, but not a clear historical BUY/HOLD/SELL analyst recommendation ground-truth endpoint.",
        "use_case": "Supplemental data, not primary recommendation ground truth.",
    },
]


def normalize_recommendation(value: Any) -> Optional[str]:
    text = str(value or "").strip().upper().replace("_", " ")
    if not text:
        return None
    buy_terms = {"BUY", "STRONG BUY", "OUTPERFORM", "OVERWEIGHT", "ACCUMULATE", "POSITIVE"}
    hold_terms = {"HOLD", "NEUTRAL", "MARKET PERFORM", "EQUAL WEIGHT", "SECTOR PERFORM", "PERFORM"}
    sell_terms = {"SELL", "STRONG SELL", "UNDERPERFORM", "UNDERWEIGHT", "REDUCE", "NEGATIVE"}
    if text in buy_terms:
        return "BUY"
    if text in hold_terms:
        return "HOLD"
    if text in sell_terms:
        return "SELL"
    if "BUY" in text or "OUTPERFORM" in text or "OVERWEIGHT" in text:
        return "BUY"
    if "SELL" in text or "UNDERPERFORM" in text or "UNDERWEIGHT" in text:
        return "SELL"
    if "HOLD" in text or "NEUTRAL" in text or "PERFORM" in text:
        return "HOLD"
    return None


def _first_present(row: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in row and row.get(key) not in (None, ""):
            return row.get(key)
    return None


def _normalize_ground_truth_row(row: Dict[str, Any], source_hint: str = "external") -> Optional[Dict[str, Any]]:
    ticker = _first_present(row, ["ticker", "Ticker", "symbol", "Symbol"])
    date = _first_present(row, ["date", "Date", "as_of_date", "rating_date", "publishedDate", "published_date"])
    rating = _first_present(row, ["recommendation", "Recommendation", "rating", "Rating", "newRating", "new_rating", "consensus", "grade"])
    normalized = normalize_recommendation(rating)
    if not ticker or not date or not normalized:
        return None
    parsed_date = pd.to_datetime(date, errors="coerce", utc=True)
    if pd.isna(parsed_date):
        return None
    return {
        "ticker": str(ticker).upper().strip(),
        "date": parsed_date,
        "recommendation": normalized,
        "raw_recommendation": str(rating),
        "source": str(row.get("source") or row.get("Source") or source_hint),
    }


def load_external_ground_truth(paths: Optional[List[str]] = None) -> Dict[str, Any]:
    paths = paths or DEFAULT_GROUND_TRUTH_PATHS
    rows: List[Dict[str, Any]] = []
    loaded_paths: List[str] = []
    issues: List[str] = []

    for path in paths:
        if not path or not os.path.exists(path):
            continue
        loaded_paths.append(path)
        try:
            if path.lower().endswith(".csv"):
                with open(path, "r", encoding="utf-8-sig", newline="") as file:
                    for raw in csv.DictReader(file):
                        item = _normalize_ground_truth_row(raw, os.path.basename(path))
                        if item:
                            rows.append(item)
            elif path.lower().endswith(".jsonl"):
                with open(path, "r", encoding="utf-8") as file:
                    for line in file:
                        try:
                            raw = json.loads(line)
                        except Exception:
                            continue
                        item = _normalize_ground_truth_row(raw, os.path.basename(path))
                        if item:
                            rows.append(item)
            else:
                issues.append(f"Unsupported ground-truth file type: {path}")
        except Exception as exc:
            issues.append(f"{path}: {exc}")

    if not rows:
        return {
            "status": "UNAVAILABLE",
            "rows": [],
            "loaded_paths": loaded_paths,
            "issues": issues or ["No external ground-truth recommendation file found."],
            "expected_schema": {
                "required": ["Date/date", "Ticker/symbol", "Recommendation/rating/newRating/consensus"],
                "recommendation_values": ["BUY", "HOLD", "SELL", "Strong Buy", "Neutral", "Underperform"],
            },
            "source_research": SOURCE_RESEARCH,
        }

    df = pd.DataFrame(rows).sort_values(["ticker", "date"]).reset_index(drop=True)
    return {
        "status": "SUCCESS",
        "rows": df.to_dict("records"),
        "loaded_paths": loaded_paths,
        "issues": issues,
        "source_research": SOURCE_RESEARCH,
    }


def _latest_external_recommendation(rows: List[Dict[str, Any]], ticker: str, decision_date: Any) -> Optional[Dict[str, Any]]:
    cutoff = pd.to_datetime(decision_date, errors="coerce", utc=True)
    if pd.isna(cutoff):
        return None
    ticker_norm = str(ticker).upper().strip()
    candidates = [
        row for row in rows
        if row.get("ticker") == ticker_norm and pd.to_datetime(row.get("date"), utc=True) <= cutoff
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: pd.to_datetime(item["date"], utc=True))[-1]


def evaluate_recommendation_accuracy(
    decision_rows: List[Dict[str, Any]],
    *,
    ground_truth_rows: Optional[List[Dict[str, Any]]] = None,
    ground_truth_paths: Optional[List[str]] = None,
    persist: bool = False,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    loaded = {"status": "SUCCESS", "rows": ground_truth_rows or [], "issues": [], "source_research": SOURCE_RESEARCH}
    if ground_truth_rows is None:
        loaded = load_external_ground_truth(ground_truth_paths)

    external_rows = loaded.get("rows") or []
    if not external_rows:
        return {
            "status": "UNAVAILABLE",
            "ground_truth_source": "UNAVAILABLE",
            "message": "External recommendation ground truth is not configured. Accuracy is not fabricated.",
            "total_predictions": len(decision_rows),
            "evaluated_predictions": 0,
            "historical_recommendations_evaluated": 0,
            "correct_recommendations": 0,
            "incorrect_recommendations": 0,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "confusion_matrix": [],
            "version_comparison": {
                "status": "UNAVAILABLE",
                "current_agent_accuracy": None,
                "previous_agent_accuracy": None,
                "accuracy_delta": None,
                "improved": "UNAVAILABLE",
                "reason": "External ground truth and previous-version records are required.",
            },
            "source_research": loaded.get("source_research") or SOURCE_RESEARCH,
            "configuration": {
                "accepted_files": DEFAULT_GROUND_TRUTH_PATHS,
                "loaded_paths": loaded.get("loaded_paths", []),
                "issues": loaded.get("issues", []),
            },
        }

    evaluated: List[Dict[str, Any]] = []
    missing = 0
    for row in decision_rows:
        agent_decision = normalize_recommendation(row.get("decision"))
        if not agent_decision:
            continue
        match = _latest_external_recommendation(external_rows, row.get("ticker"), row.get("as_of_date") or row.get("date"))
        if not match:
            missing += 1
            continue
        external_decision = match.get("recommendation")
        correct = agent_decision == external_decision
        evaluated.append(
            {
                "date": row.get("as_of_date") or row.get("date"),
                "ticker": row.get("ticker"),
                "agent_decision": agent_decision,
                "external_decision": external_decision,
                "external_source": match.get("source"),
                "external_date": pd.to_datetime(match.get("date"), utc=True).date().isoformat(),
                "confidence": row.get("confidence"),
                "regime": row.get("regime"),
                "sector": row.get("sector"),
                "correct": correct,
            }
        )

    if not evaluated:
        return {
            "status": "UNAVAILABLE",
            "ground_truth_source": "CONFIGURED_BUT_NO_OVERLAP",
            "message": "External recommendation records exist, but none overlap with this backtest's tickers/dates.",
            "total_predictions": len(decision_rows),
            "evaluated_predictions": 0,
            "historical_recommendations_evaluated": 0,
            "missing_ground_truth_matches": missing,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "confusion_matrix": [],
            "version_comparison": {
                "status": "UNAVAILABLE",
                "current_agent_accuracy": None,
                "previous_agent_accuracy": None,
                "accuracy_delta": None,
                "improved": "UNAVAILABLE",
                "reason": "No overlapping external labels for this run.",
            },
            "source_research": loaded.get("source_research") or SOURCE_RESEARCH,
        }

    df = pd.DataFrame(evaluated)
    correct_count = int(df["correct"].sum())
    total = int(len(df))

    def grouped_accuracy(field: str) -> List[Dict[str, Any]]:
        if field not in df.columns:
            return []
        out = []
        for key, group in df.groupby(field, dropna=False):
            out.append(
                {
                    field: str(key),
                    "count": int(len(group)),
                    "correct": int(group["correct"].sum()),
                    "accuracy": float(group["correct"].mean()),
                }
            )
        return out

    df["month"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").astype(str)
    confidence_numeric = pd.to_numeric(df["confidence"], errors="coerce")
    confidence_accuracy_correlation = None
    if confidence_numeric.notna().sum() >= 3 and df["correct"].nunique() > 1:
        confidence_accuracy_correlation = float(confidence_numeric.corr(df["correct"].astype(int)))

    external_sequence = df.sort_values(["ticker", "date"])
    stability_rows = []
    for ticker, group in external_sequence.groupby("ticker"):
        changes = int((group["external_decision"] != group["external_decision"].shift(1)).sum() - 1) if len(group) > 1 else 0
        stability_rows.append(
            {
                "ticker": ticker,
                "observations": int(len(group)),
                "recommendation_changes": max(0, changes),
                "stability": 1.0 - (max(0, changes) / max(1, len(group) - 1)) if len(group) > 1 else 1.0,
            }
        )

    labels = ["BUY", "HOLD", "SELL"]
    confusion_rows: List[Dict[str, Any]] = []
    per_class: List[Dict[str, Any]] = []
    for actual in labels:
        actual_mask = df["external_decision"] == actual
        for predicted in labels:
            confusion_rows.append(
                {
                    "actual": actual,
                    "predicted": predicted,
                    "count": int(((df["external_decision"] == actual) & (df["agent_decision"] == predicted)).sum()),
                }
            )
        tp = int(((df["external_decision"] == actual) & (df["agent_decision"] == actual)).sum())
        fp = int(((df["external_decision"] != actual) & (df["agent_decision"] == actual)).sum())
        fn = int(((df["external_decision"] == actual) & (df["agent_decision"] != actual)).sum())
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        f1 = (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and (precision + recall) else None
        per_class.append(
            {
                "class": actual,
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    macro_precision = float(pd.Series([row["precision"] for row in per_class if row["precision"] is not None]).mean()) if any(row["precision"] is not None for row in per_class) else None
    macro_recall = float(pd.Series([row["recall"] for row in per_class if row["recall"] is not None]).mean()) if any(row["recall"] is not None for row in per_class) else None
    macro_f1 = float(pd.Series([row["f1"] for row in per_class if row["f1"] is not None]).mean()) if any(row["f1"] is not None for row in per_class) else None

    result = {
        "status": "SUCCESS",
        "ground_truth_source": "EXTERNAL_RECOMMENDATION_RECORDS",
        "total_predictions": len(decision_rows),
        "evaluated_predictions": total,
        "historical_recommendations_evaluated": total,
        "correct_recommendations": correct_count,
        "incorrect_recommendations": total - correct_count,
        "accuracy": correct_count / total if total else None,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1,
        "per_class_metrics": per_class,
        "confusion_matrix": confusion_rows,
        "version_comparison": {
            "status": "CURRENT_ONLY",
            "current_agent_accuracy": correct_count / total if total else None,
            "previous_agent_accuracy": None,
            "accuracy_delta": None,
            "improved": "UNAVAILABLE",
            "reason": "Current run is evaluated; previous agent version records are not configured.",
        },
        "buy_accuracy": next((x["accuracy"] for x in grouped_accuracy("agent_decision") if x["agent_decision"] == "BUY"), None),
        "hold_accuracy": next((x["accuracy"] for x in grouped_accuracy("agent_decision") if x["agent_decision"] == "HOLD"), None),
        "sell_accuracy": next((x["accuracy"] for x in grouped_accuracy("agent_decision") if x["agent_decision"] == "SELL"), None),
        "confidence_accuracy_correlation": confidence_accuracy_correlation,
        "recommendation_stability": stability_rows,
        "rolling_monthly_accuracy": grouped_accuracy("month"),
        "ticker_accuracy": grouped_accuracy("ticker"),
        "sector_accuracy": grouped_accuracy("sector"),
        "regime_accuracy": grouped_accuracy("regime"),
        "evaluated_rows": evaluated,
        "missing_ground_truth_matches": missing,
        "source_research": loaded.get("source_research") or SOURCE_RESEARCH,
    }

    if persist:
        payload = {
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "accuracy": {k: v for k, v in result.items() if k != "evaluated_rows"},
        }
        append_jsonl(RECOMMENDATION_ACCURACY_PATH, payload)
    return result


def evaluate_feature_versions(
    feature_runs: Dict[str, List[Dict[str, Any]]],
    *,
    ground_truth_rows: Optional[List[Dict[str, Any]]] = None,
    ground_truth_paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    rows = []
    for feature_name, decisions in feature_runs.items():
        accuracy = evaluate_recommendation_accuracy(
            decisions,
            ground_truth_rows=ground_truth_rows,
            ground_truth_paths=ground_truth_paths,
            persist=False,
        )
        rows.append(
            {
                "version": feature_name,
                "status": accuracy.get("status"),
                "total_predictions": accuracy.get("total_predictions"),
                "evaluated_predictions": accuracy.get("evaluated_predictions"),
                "accuracy": accuracy.get("accuracy"),
                "precision": accuracy.get("precision"),
                "recall": accuracy.get("recall"),
                "f1": accuracy.get("f1"),
            }
        )
    successful = [row for row in rows if row.get("status") == "SUCCESS" and row.get("accuracy") is not None]
    baseline = successful[0] if successful else None
    for row in rows:
        row["accuracy_delta_vs_baseline"] = (
            float(row["accuracy"]) - float(baseline["accuracy"])
            if baseline and row.get("accuracy") is not None
            else None
        )
    return {
        "status": "SUCCESS" if successful else "UNAVAILABLE",
        "rows": rows,
        "best_version": max(successful, key=lambda item: item["accuracy"]) if successful else None,
        "baseline_version": baseline,
        "message": "Feature versions are compared only when external ground truth is available.",
    }
