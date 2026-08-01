"""
Strategy Calibration Engine v3
Manages calibration dataset for the AI strategy prediction surrogate.
Runs mini tastytrade backtests on similar windows to build a corpus of
empirical ground-truth data, allowing the AI to learn strategy-level
calibration factors without copying the exact target backtest result.

NO external price APIs. NO random values. NO backtest result leakage.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from jsonl_store import append_jsonl

ROOT          = Path(__file__).resolve().parent
EVAL_FILE     = ROOT / "strategy_prediction_evaluation_runs.jsonl"
CALIB_SOURCE  = "strategy_calibration_backtest"
EVAL_SOURCE   = "strategy_backtest_verification"
MODEL_VERSION = "strategy_predictor_v3_backtest_surrogate_calibrated"
MIN_SIMILARITY = 0.50

_HASH_FIELDS = [
    "symbol", "start_date", "end_date", "initial_capital",
    "benchmark", "direction", "side", "dte", "delta",
    "legs", "entry_frequency", "decision_horizon",
]


# ══════════════════════════════════════════════════════════════════════════════
# HASH
# ══════════════════════════════════════════════════════════════════════════════
def _build_strategy_hash(si: dict) -> str:
    """SHA-256[:16] of canonical sorted-key JSON — identical to strategy_prediction_agent."""
    import hashlib
    subset = {k: si[k] for k in sorted(_HASH_FIELDS) if k in si}
    return hashlib.sha256(json.dumps(subset, sort_keys=True).encode()).hexdigest()[:16]


# ══════════════════════════════════════════════════════════════════════════════
# BUCKET HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _delta_bucket(d: int) -> int:
    return 0 if d <= 20 else 1 if d <= 35 else 2 if d <= 50 else 3


def _dte_bucket(d: int) -> int:
    return 0 if d <= 20 else 1 if d <= 35 else 2 if d <= 60 else 3


def _freq_bucket(f: str) -> int:
    f = str(f).lower()
    return 0 if "day" in f else 1 if "week" in f else 2


def _date_regime_bucket(start_date: str) -> int:
    """Regime based on start year: 0=pre-2015, 1=2015-2018, 2=2019-2021, 3=2022+."""
    try:
        year = int(str(start_date)[:4])
    except (ValueError, TypeError):
        return 1
    if year < 2015: return 0
    if year < 2019: return 1
    if year < 2022: return 2
    return 3


# ══════════════════════════════════════════════════════════════════════════════
# SIMILARITY
# ══════════════════════════════════════════════════════════════════════════════
def compute_strategy_similarity(si_a: dict, si_b: dict) -> float:
    """
    Strategy similarity in [0.0, 1.0].
    Weights: symbol 0.15, direction 0.15, side 0.15, delta_bucket 0.15,
             dte_bucket 0.10, legs 0.10, entry_frequency 0.10,
             date_regime 0.05, benchmark 0.05  (total = 1.00).
    Threshold for calibration use: MIN_SIMILARITY = 0.50.
    """
    score = 0.0

    if str(si_a.get("symbol",         "")).upper() == str(si_b.get("symbol",         "")).upper():
        score += 0.15
    if str(si_a.get("direction",      "")).lower() == str(si_b.get("direction",      "")).lower():
        score += 0.15
    if str(si_a.get("side",           "")).lower() == str(si_b.get("side",           "")).lower():
        score += 0.15
    if _delta_bucket(int(si_a.get("delta",           30))) == _delta_bucket(int(si_b.get("delta",           30))):
        score += 0.15
    if _dte_bucket(  int(si_a.get("dte",             45))) == _dte_bucket(  int(si_b.get("dte",             45))):
        score += 0.10
    if int(si_a.get("legs",           1)) == int(si_b.get("legs",           1)):
        score += 0.10
    if _freq_bucket(si_a.get("entry_frequency", "monthly")) == _freq_bucket(si_b.get("entry_frequency", "monthly")):
        score += 0.10
    if _date_regime_bucket(si_a.get("start_date", "2020-01-01")) == _date_regime_bucket(si_b.get("start_date", "2020-01-01")):
        score += 0.05
    if str(si_a.get("benchmark", "SPY")).upper() == str(si_b.get("benchmark", "SPY")).upper():
        score += 0.05

    return round(score, 3)


# ══════════════════════════════════════════════════════════════════════════════
# RECORD LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_similar_records(
    strategy_input: dict,
    current_hash: str,
    min_similarity: float = MIN_SIMILARITY,
) -> List[Tuple[float, dict]]:
    """
    Load ALL records from EVAL_FILE with valid backtest_actual data and
    similarity >= min_similarity to strategy_input.

    Accepts:
    - source == strategy_backtest_verification (any model_version — backtest_actual is tastytrade)
    - source == strategy_calibration_backtest (mini probes, any model_version)

    Excludes:
    - current_hash (no leakage from the very run being predicted)
    - records with failed/null backtests (decision=REVIEW or no return data)

    Returns: list of (similarity_score, record) sorted desc by similarity.
    """
    results: List[Tuple[float, dict]] = []
    if not EVAL_FILE.exists():
        return results

    try:
        with open(EVAL_FILE, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue

                source = r.get("source", "")
                if source not in (EVAL_SOURCE, CALIB_SOURCE):
                    continue

                if current_hash and r.get("strategy_input_hash") == current_hash:
                    continue

                bt = r.get("backtest_actual", {}) or {}
                if not bt:
                    continue

                has_return   = (bt.get("actual_total_return_pct") is not None or
                                bt.get("total_return_pct") is not None)
                has_decision = bt.get("decision") not in (None, "REVIEW", "MISSING", "")
                has_passed   = bool(bt.get("passed_validation", False))

                if not (has_passed or (has_return and has_decision)):
                    continue

                sim = compute_strategy_similarity(strategy_input, r.get("strategy_input", {}))
                if sim >= min_similarity:
                    results.append((sim, r))

    except Exception:
        pass

    return sorted(results, key=lambda x: x[0], reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# CALIBRATION WINDOW GENERATION
# ══════════════════════════════════════════════════════════════════════════════
def _generate_calibration_windows(
    strategy_input: dict,
    current_hash: str,
    n: int = 14,
) -> List[dict]:
    """
    Generate up to n distinct sub-windows (different date ranges) for the
    same strategy type.  None share the same hash as current_hash.
    Windows are at least 180 days long.
    """
    try:
        start_dt = datetime.strptime(str(strategy_input["start_date"]), "%Y-%m-%d")
        end_dt   = datetime.strptime(str(strategy_input["end_date"]),   "%Y-%m-%d")
    except (ValueError, KeyError):
        return []

    total_days = (end_dt - start_dt).days
    if total_days < 365:
        return []

    base_si = {k: v for k, v in strategy_input.items()
               if k not in ("start_date", "end_date")}

    windows: List[dict] = []
    seen_hashes = {current_hash}
    seen_windows: set = set()

    def _try_add(w_start: datetime, w_end: datetime) -> bool:
        if (w_end - w_start).days < 180:
            return False
        w_key = (w_start.strftime("%Y-%m-%d"), w_end.strftime("%Y-%m-%d"))
        if w_key in seen_windows:
            return False
        w_si = {**base_si,
                "start_date": w_start.strftime("%Y-%m-%d"),
                "end_date":   w_end.strftime("%Y-%m-%d")}
        w_hash = _build_strategy_hash(w_si)
        if w_hash in seen_hashes:
            return False
        seen_hashes.add(w_hash)
        seen_windows.add(w_key)
        windows.append(w_si)
        return True

    # 12-month rolling windows, 6-month step
    for offset in range(0, total_days // 30, 6):
        if len(windows) >= n:
            break
        w_start = start_dt + timedelta(days=offset * 30)
        w_end   = min(w_start + timedelta(days=365), end_dt)
        _try_add(w_start, w_end)

    # 18-month windows, 9-month step
    for offset in range(0, total_days // 30, 9):
        if len(windows) >= n:
            break
        w_start = start_dt + timedelta(days=offset * 30)
        w_end   = min(w_start + timedelta(days=548), end_dt)
        _try_add(w_start, w_end)

    # 24-month windows, 12-month step
    for offset in range(0, total_days // 30, 12):
        if len(windows) >= n:
            break
        w_start = start_dt + timedelta(days=offset * 30)
        w_end   = min(w_start + timedelta(days=730), end_dt)
        _try_add(w_start, w_end)

    return windows[:n]


# ══════════════════════════════════════════════════════════════════════════════
# CALIBRATION DATASET MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════
def _save_calibration_record(record: dict) -> None:
    """Append one calibration probe record to EVAL_FILE."""
    try:
        append_jsonl(EVAL_FILE, record)
    except Exception:
        pass


def ensure_calibration_dataset(
    strategy_input: dict,
    current_hash: str,
    min_records: int = 8,
    max_new_backtests: int = 12,
    backtest_fn: Optional[Callable] = None,
) -> List[Tuple[float, dict]]:
    """
    Ensure we have >= min_records similar calibration records.

    If fewer exist, runs mini tastytrade backtests on different time windows
    of the same strategy type (NOT the exact current full window) and saves
    them as source=strategy_calibration_backtest.

    Saved records are immediately available for the AI calibration step.

    Args:
        strategy_input: canonical StrategyInput dict
        current_hash:   SHA-256[:16] of current run — excluded from calibration
        min_records:    target number of similar records before we stop adding
        max_new_backtests: hard cap on new mini backtests in one call
        backtest_fn:    callable(si: dict) -> dict (run_tastytrade_strategy_backtest)
                        None = read-only mode, no new backtests

    Returns:
        list of (similarity_score, record) tuples, sorted desc by similarity
    """
    records = load_similar_records(strategy_input, current_hash)

    if len(records) >= min_records or backtest_fn is None:
        return records

    existing_hashes = {r.get("strategy_input_hash") for _, r in records}

    windows = _generate_calibration_windows(
        strategy_input, current_hash, n=max_new_backtests * 2
    )

    new_count = 0
    for w_si in windows:
        if len(records) >= min_records or new_count >= max_new_backtests:
            break

        w_hash = _build_strategy_hash(w_si)
        if w_hash in existing_hashes:
            continue

        try:
            bt = backtest_fn(w_si)
        except Exception:
            continue

        if not bt:
            continue

        has_passed = bool(bt.get("passed_validation", False))
        has_return = (bt.get("total_return_pct") is not None or
                      bt.get("actual_total_return_pct") is not None)

        if not (has_passed and has_return):
            continue

        calib_rec = {
            "timestamp":               datetime.utcnow().isoformat() + "Z",
            "source":                  CALIB_SOURCE,
            "model_version":           MODEL_VERSION,
            "strategy_input":          {k: w_si.get(k) for k in [
                "symbol", "start_date", "end_date", "initial_capital",
                "benchmark", "direction", "side", "dte", "delta",
                "legs", "entry_frequency",
            ]},
            "strategy_input_hash":     w_hash,
            "created_for_parent_hash": current_hash,
            "backtest_actual":         bt,
            "calibration_record":      True,
        }
        _save_calibration_record(calib_rec)
        existing_hashes.add(w_hash)

        sim = compute_strategy_similarity(strategy_input, w_si)
        records.append((sim, calib_rec))
        new_count += 1

    return sorted(records, key=lambda x: x[0], reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# CALIBRATION MODEL FITTING
# ══════════════════════════════════════════════════════════════════════════════
def fit_calibration_model(
    strategy_input: dict,
    raw_prediction: dict,
    cal_records_with_similarity: List[Tuple[float, dict]],
) -> dict:
    """
    Learn calibration factors from empirical backtest records.

    Returns a calibration model dict:
    - has_calibration: bool
    - n_records: int
    - empirical_return: weighted avg actual return %
    - empirical_win_rate: weighted avg actual win rate %
    - empirical_max_drawdown: weighted avg actual max drawdown %
    - calibration_weight / raw_weight: blending weights per count schedule
    - return_scale: actual / raw for reference (clamped)
    - empirical_directional: "positive" / "negative" / "neutral"
    """
    n = len(cal_records_with_similarity)
    if n < 2:
        return {
            "has_calibration":   False,
            "n_records":         n,
            "calibration_weight": 0.0,
            "raw_weight":         1.0,
        }

    total_sim = sum(s for s, _ in cal_records_with_similarity)
    if total_sim == 0:
        return {"has_calibration": False, "n_records": n,
                "calibration_weight": 0.0, "raw_weight": 1.0}

    def _wavg(key_fn) -> Optional[float]:
        t = 0.0; tw = 0.0
        for sim, r in cal_records_with_similarity:
            bt  = r.get("backtest_actual", {}) or {}
            val = key_fn(bt)
            if val is not None:
                try:
                    t += sim * float(val); tw += sim
                except (TypeError, ValueError):
                    pass
        return round(t / tw, 3) if tw > 0 else None

    emp_ret = _wavg(lambda bt: bt.get("actual_total_return_pct") or bt.get("total_return_pct"))
    emp_wr  = _wavg(lambda bt: bt.get("actual_win_rate")         or bt.get("win_rate"))
    emp_dd  = _wavg(lambda bt: bt.get("actual_max_drawdown")     or bt.get("max_drawdown"))

    # Blending weight schedule per spec
    if n >= 8:   w_cal, w_raw = 0.75, 0.25
    elif n >= 5: w_cal, w_raw = 0.60, 0.40
    elif n >= 3: w_cal, w_raw = 0.45, 0.55
    else:        w_cal, w_raw = 0.20, 0.80

    # Reference: return_scale (how much raw model undershoots vs empirical)
    raw_ret = raw_prediction.get("predicted_total_return_pct", 0) if raw_prediction else 0
    try:
        if raw_ret and raw_ret != 0 and emp_ret is not None:
            ret_scale = round(max(0.1, min(50.0, emp_ret / raw_ret)), 3)
        else:
            ret_scale = None
    except (TypeError, ZeroDivisionError):
        ret_scale = None

    # Directional vote from empirical records
    decisions = [
        str((r.get("backtest_actual", {}) or {}).get("decision", "REVIEW")).upper()
        for _, r in cal_records_with_similarity
    ]
    buy_ct  = decisions.count("BUY")
    sell_ct = decisions.count("SELL")
    hold_ct = decisions.count("HOLD")
    emp_dir = ("positive" if buy_ct >= max(sell_ct, hold_ct) else
               ("negative" if sell_ct > max(buy_ct, hold_ct) else "neutral"))

    return {
        "has_calibration":       True,
        "n_records":             n,
        "empirical_return":      emp_ret,
        "empirical_win_rate":    emp_wr,
        "empirical_max_drawdown": emp_dd,
        "calibration_weight":    w_cal,
        "raw_weight":            w_raw,
        "return_scale":          ret_scale,
        "empirical_directional": emp_dir,
        "buy_count":             buy_ct,
        "sell_count":            sell_ct,
        "hold_count":            hold_ct,
    }
