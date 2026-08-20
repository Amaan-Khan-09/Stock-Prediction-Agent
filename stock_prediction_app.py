"""
AI Financial Analyst System -- One-Month Prediction Validation

Two modes:
  1. Stock Price Validation  : AI predicts stock price movement vs actual historical prices
  2. Options Strategy Validation : AI predicts + tastytrade options backtest (prediction window only)

TWO-WINDOW ARCHITECTURE:
  Historical Context Window   : historical_context_start_date -> prediction_origin_date  (AI study only)
  Prediction/Validation Window: prediction_origin_date -> target_date                    (AI predicts; validation checks)

Options backtest ONLY runs for prediction/validation window -- NEVER for historical context window.

Run: streamlit run stock_prediction_app.py
"""
from __future__ import annotations

import os
import sys
import time
import uuid
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "tools"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)

import streamlit as st

st.set_page_config(
    page_title="AI Financial Analyst System - Stock Prediction",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_data(ttl=300)
def _get_symbol_accuracy_stats(symbol: str) -> dict:
    """Load per-symbol accuracy stats from evaluation runs (cached 5 min)."""
    import json as _json_sas
    _LOW_ACC = {"META": 0, "GOOGL": 0, "NFLX": 0, "UNH": 0, "TSLA": 17}
    sym_upper = symbol.strip().upper()
    result = {"symbol": sym_upper, "runs": 0, "match_pct": None, "warning": None, "is_blacklisted": False}
    try:
        eval_path = ROOT / "stock_prediction_evaluation_runs.jsonl"
        if not eval_path.exists():
            if sym_upper in _LOW_ACC:
                result["match_pct"] = _LOW_ACC[sym_upper]
                result["warning"] = "critical" if _LOW_ACC[sym_upper] == 0 else "high"
                result["is_blacklisted"] = _LOW_ACC[sym_upper] == 0
            return result
        records = []
        with open(eval_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(_json_sas.loads(line))
                    except Exception:
                        pass
        sym_recs = [r for r in records if
                    str((r.get("stock_prediction_input") or r.get("spi") or {}).get("symbol", "")).upper() == sym_upper]
        if len(sym_recs) >= 3:
            match = sum(1 for r in sym_recs if r.get("comparison", {}).get("decision_match"))
            match_pct = round(match / len(sym_recs) * 100, 0)
            result["runs"] = len(sym_recs)
            result["match_pct"] = match_pct
            if match_pct == 0:
                result["warning"] = "critical"
                result["is_blacklisted"] = True
            elif match_pct < 20 and len(sym_recs) >= 5:
                result["warning"] = "critical"
                result["is_blacklisted"] = True
                result["blacklisted"] = True
            elif match_pct < 25:
                result["warning"] = "high"
            elif match_pct < 40:
                result["warning"] = "medium"
        elif sym_upper in _LOW_ACC:
            result["match_pct"] = _LOW_ACC[sym_upper]
            result["warning"] = "critical" if _LOW_ACC[sym_upper] == 0 else "high"
            result["is_blacklisted"] = _LOW_ACC[sym_upper] == 0
            result["blacklisted"] = _LOW_ACC[sym_upper] == 0
    except Exception:
        pass
    return result


def _render_symbol_accuracy_warning(symbol: str):
    """Render a warning banner if symbol has low AI accuracy history."""
    if not symbol or len(symbol) < 1:
        return
    stats = _get_symbol_accuracy_stats(symbol.strip().upper())
    sym_upper = symbol.strip().upper()

    # Phase 6 — Auto-blacklist: accuracy < 20% AND runs >= 5
    if stats.get("blacklisted") and stats.get("runs", 0) >= 5:
        _acc   = int(stats.get("match_pct", 0))
        _runs  = stats.get("runs", 0)
        st.error(
            f"UNRELIABLE SYMBOL — AI has {_acc}% accuracy on {sym_upper} across {_runs} runs. "
            f"Historical failure pattern: near-zero direction accuracy. "
            f"Proceeding is NOT recommended. Results will likely be inaccurate."
        )
        st.warning(
            "If you still want to proceed, the system will apply maximum calibration corrections "
            "but accuracy cannot be guaranteed."
        )
        return

    if stats["warning"] == "critical":
        st.markdown(
            f'<div style="background:#1C0A0A;border:2px solid #EF4444;border-radius:8px;padding:.6rem 1rem;margin:.4rem 0">'
            f'<span style="color:#EF4444;font-weight:900;font-size:.85rem">AI ACCURACY WARNING -- {sym_upper}</span><br>'
            f'<span style="color:#FCA5A5;font-size:.8rem">'
            f'Historical accuracy for <b>{sym_upper}</b>: '
            f'<b>{int(stats["match_pct"])}% decision match</b> across {stats["runs"]} runs. '
            f'This symbol has a near-zero AI accuracy record. '
            f'AI predictions for {sym_upper} are highly unreliable -- treat all outputs as exploratory only. '
            f'Consider using SPY or SPX for more reliable results.'
            f'</span></div>',
            unsafe_allow_html=True,
        )
    elif stats["warning"] == "high":
        st.markdown(
            f'<div style="background:#1C1000;border:1.5px solid #F59E0B;border-radius:8px;padding:.5rem 1rem;margin:.4rem 0">'
            f'<span style="color:#F59E0B;font-weight:800;font-size:.82rem">LOW ACCURACY -- {sym_upper}</span><br>'
            f'<span style="color:#FCD34D;font-size:.78rem">'
            f'<b>{sym_upper}</b>: {int(stats["match_pct"])}% decision match across {stats["runs"]} runs. '
            f'AI predictions are below the 40% threshold for this symbol. Results may be unreliable.'
            f'</span></div>',
            unsafe_allow_html=True,
        )
    elif stats["warning"] == "medium":
        st.markdown(
            f'<div style="background:#0A1000;border:1px solid #84CC16;border-radius:8px;padding:.4rem 1rem;margin:.4rem 0">'
            f'<span style="color:#A3E635;font-size:.78rem">'
            f'INFO: <b>{sym_upper}</b>: {int(stats["match_pct"])}% decision match ({stats["runs"]} runs). Below 60% target but usable.'
            f'</span></div>',
            unsafe_allow_html=True,
        )
    elif stats["runs"] >= 3 and stats["match_pct"] is not None and stats["match_pct"] >= 40:
        st.markdown(
            f'<div style="background:#001A0A;border:1px solid #10B981;border-radius:8px;padding:.4rem 1rem;margin:.4rem 0">'
            f'<span style="color:#6EE7B7;font-size:.78rem">'
            f'OK: <b>{sym_upper}</b>: {int(stats["match_pct"])}% decision match ({stats["runs"]} runs). Reasonable accuracy history.'
            f'</span></div>',
            unsafe_allow_html=True,
        )


def _apply_light_demo_theme():
    """Force white/light professional UI — overrides any dark browser/OS defaults."""
    st.markdown("""
<style>
/* ── Page root ─────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"],
.stApp, section.main, .main .block-container {
    background-color: #FFFFFF !important;
    color: #111827 !important;
}
/* ── Sidebar ───────────────────────────────────────────── */
[data-testid="stSidebar"], [data-testid="stSidebarContent"] {
    background-color: #F8FAFC !important;
}
/* ── Header / toolbar ──────────────────────────────────── */
[data-testid="stHeader"] {
    background-color: #FFFFFF !important;
}
/* ── Block container ───────────────────────────────────── */
.block-container {
    background-color: #FFFFFF !important;
    padding-top: 1rem !important;
}
/* ── Inputs ────────────────────────────────────────────── */
.stTextInput input, .stNumberInput input,
.stDateInput input, .stSelectbox select,
.stTextArea textarea {
    background-color: #FFFFFF !important;
    color: #111827 !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 6px !important;
}
/* ── Selectbox / dropdown ──────────────────────────────── */
.stSelectbox > div > div {
    background-color: #FFFFFF !important;
    color: #111827 !important;
    border: 1px solid #D1D5DB !important;
}
/* ── Radio buttons ─────────────────────────────────────── */
.stRadio > div { background-color: #FFFFFF !important; }
/* ── Buttons ───────────────────────────────────────────── */
.stButton > button {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
}
.stButton > button:hover {
    background-color: #1D4ED8 !important;
}
/* ── Expanders ─────────────────────────────────────────── */
.streamlit-expanderHeader, .streamlit-expanderContent,
[data-testid="stExpander"] {
    background-color: #F8FAFC !important;
    color: #111827 !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 6px !important;
}
/* ── Metrics ───────────────────────────────────────────── */
[data-testid="metric-container"] {
    background-color: #F8FAFC !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 8px !important;
    padding: 0.5rem !important;
}
/* ── Dataframes / tables ───────────────────────────────── */
.stDataFrame, [data-testid="stTable"] {
    background-color: #FFFFFF !important;
}
/* ── Info / warning / error boxes ─────────────────────── */
.stAlert {
    background-color: #F8FAFC !important;
    color: #111827 !important;
}
/* ── Text / labels (light backgrounds only) ────────────── */
label, p, .stMarkdown, h1, h2, h3 {
    color: #111827 !important;
}
/* ── Dark panel children: respect inline color styles ───── */
/* span elements inside dark panels carry explicit color attrs;
   excluding span from global !important lets them render correctly */
/* ── Vertical/Horizontal blocks ────────────────────────── */
div[data-testid="stVerticalBlock"],
div[data-testid="stHorizontalBlock"] {
    background-color: transparent !important;
}
/* ── JSON viewer ───────────────────────────────────────── */
.stJson { background-color: #F8FAFC !important; }
/* ── Spinner ───────────────────────────────────────────── */
.stSpinner { color: #2563EB !important; }
/* ── Dark panels: all inline children keep their color ─── */
/* Targets divs with dark backgrounds injected via st.markdown */
[data-testid="stMarkdownContainer"] div[style*="background:#0A"],
[data-testid="stMarkdownContainer"] div[style*="background:#05"],
[data-testid="stMarkdownContainer"] div[style*="background:#0F"] {
    color: #F8FAFC !important;
}
[data-testid="stMarkdownContainer"] div[style*="background:#0A"] *,
[data-testid="stMarkdownContainer"] div[style*="background:#05"] *,
[data-testid="stMarkdownContainer"] div[style*="background:#0F"] * {
    color: inherit;
}
</style>
""", unsafe_allow_html=True)


_apply_light_demo_theme()

# ── Stock prediction core imports ──────────────────────────────────────────────
from historical_price_service import (
    fetch_price_history,
    fetch_price_history_for_range,
    filter_history_up_to,
    get_context_summary,
    get_provider_used,
    HISTORICAL_PRICE_PROVIDER,
)
from stock_prediction_agent import (
    run_stock_prediction,
    build_stock_prediction_hash,
    validate_stock_prediction_input,
    MODEL_VERSION,
)
from gemini_stock_prediction_agent import (
    run_gemini_stock_prediction,
    build_stock_feature_packet,
    AI_PROVIDER as _AI_PROVIDER_CFG,
    ALLOW_BASELINE_FALLBACK as _ALLOW_BASELINE_FALLBACK,
    GEMINI_MODEL as _GEMINI_MODEL_CFG,
)
from stock_walkforward_validator import run_stock_validation
from stock_comparison_engine import compare_ai_vs_actual, compute_actual_metrics
from stock_accuracy_engine import (
    save_stock_prediction_record,
    load_stock_prediction_records,
    get_accuracy_summary,
)
from pending_predictions_engine import (
    save_pending_prediction,
    load_pending_predictions,
    count_pending_predictions,
)

# ── Optional tastytrade backtester (Options mode) ──────────────────────────────
try:
    from src.services.tastytrade_backtester_service import (
        create_backtest as _tt_create_backtest,
        poll_backtest as _tt_poll_backtest,
        build_custom_legs_payload as _tt_build_legs,
    )
    _TT_AVAILABLE = True
except Exception:
    _TT_AVAILABLE = False

# ── Paid API health-check services ─────────────────────────────────────────────
try:
    from rapidapi_market_service import run_rapidapi_market_health_check as _rapidapi_hc
    _RAPIDAPI_SVC_AVAILABLE = True
except Exception:
    _RAPIDAPI_SVC_AVAILABLE = False
    def _rapidapi_hc(**kwargs):
        return {
            "called": False, "endpoint": "", "http_status": 0,
            "total_count": None, "top_symbols": [], "key_present": False,
            "role": "market_intelligence_health_check",
            "used_in_prediction_context": False,
            "error": "rapidapi_market_service import failed",
        }

try:
    from tastytrade_service import run_tastytrade_health_check as _tastytrade_hc
    _TASTYTRADE_SVC_AVAILABLE = True
except Exception:
    _TASTYTRADE_SVC_AVAILABLE = False
    def _tastytrade_hc():
        return {
            "called": False, "endpoint": "", "http_status": 0, "latency_ms": None,
            "token_refreshed": False, "customer_verified": False,
            "refresh_present": False,
            "role": "authentication_and_account_health_check",
            "used_in_stock_mode": False, "used_in_options_mode": True,
            "error": "tastytrade_service import failed",
        }


# ── Session keys cleared on every new run ────────────────────────────────────
_RUN_KEYS = (
    "spi", "price_hist", "ctx_summary", "ai_result", "val_result",
    "opts_result", "opts_params", "backtest_payload",
    "saved", "save_msg", "input_hash", "run_id", "run_ts",
    "error_msg", "active_mode", "run_type",
    "rapidapi_health", "tastytrade_health", "run_context",
    "ctx_optimizer_results",
)


def _clear_run():
    for k in _RUN_KEYS:
        st.session_state.pop(k, None)


def _log_invalid_run(
    run_id: str, input_hash: str, symbol: str, mode: str,
    failure_reason: str, provider_error: str,
) -> None:
    """Append a failed/blocked run to the quarantine log — never the accuracy log."""
    import json as _json
    import datetime as _dt
    log_path = Path(__file__).parent / "data" / "invalid_runs.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "run_id": run_id,
        "input_hash": input_hash,
        "symbol": symbol,
        "mode": mode,
        "failure_reason": failure_reason,
        "provider_error": provider_error[:500] if provider_error else None,
        "timestamp_utc": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "accuracy_saved": False,
    }
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(_json.dumps(entry) + "\n")


_BUILD_TS = __import__("datetime").datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
_BUILD_FILE = __file__


def _dispatch_prediction(spi: dict, ctx_bars: list) -> dict:
    """Route to Gemini or baseline — reads env at call time, never uses stale import."""
    # Read FRESH from environment every single call — bypasses module-level cache
    provider       = os.getenv("AI_PROVIDER", "gemini").lower().strip()
    allow_fallback = os.getenv("ALLOW_BASELINE_FALLBACK", "false").lower() == "true"

    import sys as _sys
    print(f"[DISPATCH] provider={provider!r}  allow_fallback={allow_fallback}  "
          f"symbol={spi.get('symbol')}  origin={spi.get('prediction_origin_date')}",
          flush=True, file=_sys.stderr)

    if provider == "gemini":
        result = run_gemini_stock_prediction(spi, ctx_bars)
        print(f"[DISPATCH] Gemini returned status={result.get('status')}  "
              f"source={result.get('source')}  gemini_used={result.get('gemini_used')}",
              flush=True, file=_sys.stderr)
        if result.get("status") != "SUCCESS" and allow_fallback:
            fallback = run_stock_prediction(spi, ctx_bars)
            fallback["ai_provider"]     = "baseline_fallback"
            fallback["gemini_used"]     = False
            fallback["fallback_reason"] = result.get("error", "Gemini unavailable")
            print(f"[DISPATCH] FALLBACK triggered: {fallback['fallback_reason']}", flush=True, file=_sys.stderr)
            return fallback
        return result
    # baseline mode
    result = run_stock_prediction(spi, ctx_bars)
    result["ai_provider"] = "baseline"
    result["gemini_used"] = False
    print(f"[DISPATCH] Baseline returned source={result.get('source')}", flush=True, file=_sys.stderr)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# UI PRIMITIVES
# ══════════════════════════════════════════════════════════════════════════════

def _fmt(v, prefix="", fmt=",.2f", suffix="", default="---"):
    if v is None:
        return default
    try:
        return f"{prefix}{float(v):{fmt}}{suffix}"
    except Exception:
        return str(v)


def _sign_fmt(v, fmt=",.2f", suffix=""):
    if v is None:
        return "---"
    try:
        f = float(v)
        return f"{'+'if f >= 0 else ''}{f:{fmt}}{suffix}"
    except Exception:
        return str(v)


def _decision_badge(d: str) -> str:
    _du = str(d).upper()
    c = {"BUY": "#00875A", "SELL": "#DF1B41", "HOLD": "#D97706", "REVIEW": "#7C3AED"}.get(_du, "#6B7280")
    _suffix = " ⚠ (human review required)" if _du == "REVIEW" else ""
    return f'<span style="color:{c};font-weight:800;font-size:1.1rem">{d}{_suffix}</span>'


def _card(msg, color="#1E40AF", bg="#EFF6FF", border="#3B82F6"):
    st.markdown(
        f'<div style="background:{bg};border-left:4px solid {border};'
        f'padding:.55rem 1rem;border-radius:4px;color:{color};'
        f'font-size:.78rem;margin:.25rem 0">{msg}</div>',
        unsafe_allow_html=True,
    )


def _err_card(m):  _card(m, "#991B1B", "#FEF2F2", "#DC2626")
def _info_card(m): _card(m)
def _warn_card(m): _card(m, "#78350F", "#FFFBEB", "#F59E0B")
def _ok_card(m):   _card(m, "#065F46", "#D1FAE5", "#059669")


def _friendly_ai_error(raw_err: str) -> str:
    """Convert raw technical AI error into a user-friendly message."""
    if not raw_err:
        return "AI prediction could not be completed. Please try again."
    e = str(raw_err)
    if "JSON parse" in e or "json" in e.lower() or "delimiter" in e or "Expecting" in e:
        return (
            "AI model returned a response that could not be parsed. "
            "This is a temporary Gemini API issue — please click Predict again. "
            "If this repeats, try a shorter analysis window or different symbol."
        )
    if "API_KEY" in e or "api key" in e.lower() or "GEMINI_API_KEY" in e:
        return "Gemini API key is missing or invalid. Set GEMINI_API_KEY in your .env file."
    if "timeout" in e.lower() or "timed out" in e.lower():
        return "AI model timed out. Please try again — Gemini API may be slow."
    if "quota" in e.lower() or "rate" in e.lower() or "429" in e:
        return "Gemini API rate limit reached. Wait 60 seconds and try again."
    if "RATE_LIMITED" in e:
        return "Gemini API rate limit reached. Wait 60 seconds and try again."
    # Generic fallback — hide raw technical detail
    return "AI prediction temporarily unavailable. Please try again in a moment."


def _render_failure_panel():
    """Professional run-blocked panel — replaces simple error card.
    Shows all provider/run context so user knows exactly what failed and why.
    No old results shown alongside this panel.
    """
    error_msg = st.session_state.get("error_msg", "Unknown error")
    ctx       = st.session_state.get("run_context", {})
    run_id    = ctx.get("run_id") or st.session_state.get("run_id", "---")
    ih        = ctx.get("input_hash") or st.session_state.get("input_hash", "---")
    symbol    = ctx.get("symbol", "---")
    mode      = ctx.get("mode", "---")
    ps        = ctx.get("provider_status", {})
    errs      = ctx.get("errors", [])
    ts        = ctx.get("timestamp_utc") or st.session_state.get("run_ts", "---")

    _is_input_err = str(error_msg).startswith("Input validation failed")

    if _is_input_err:
        _banner_text  = "RUN BLOCKED — INVALID INPUT — FIX THE FORM AND RESUBMIT"
        _next_actions = [
            ("Next actions", ""),
            ("&nbsp;&nbsp;1", "<b>Fix the invalid field shown above</b> (e.g. correct the date format to YYYY-MM-DD)"),
            ("&nbsp;&nbsp;2", "Make sure all dates use format: YYYY-MM-DD (e.g. 2026-05-10, not 2026-057-10)"),
            ("&nbsp;&nbsp;3", "Click <b>Run</b> again after fixing the input"),
            ("&nbsp;&nbsp;4", "Do NOT interpret any old output above as current run result"),
        ]
    else:
        _banner_text  = "RUN BLOCKED — DATA / PROVIDER NOT AVAILABLE — OLD OUTPUT CLEARED"
        _next_actions = [
            ("Next actions", ""),
            ("&nbsp;&nbsp;1", "Retry after network / provider is available"),
            ("&nbsp;&nbsp;2", "Configure a reliable historical data provider (.env)"),
            ("&nbsp;&nbsp;3", "Use another symbol with available data"),
            ("&nbsp;&nbsp;4", "Do NOT interpret any old output above as current run result"),
        ]

    st.markdown(
        f'<div style="background:#7F1D1D;padding:.6rem 1.2rem;border-radius:6px;'
        f'margin:1rem 0 .3rem 0">'
        f'<span style="color:#FEF2F2;font-weight:900;font-size:.95rem">'
        f'{_banner_text}'
        f'</span></div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            _table([
                ("Run ID",          run_id),
                ("Input Hash",      f'<code>{ih[:16]}…</code>' if len(str(ih)) > 16 else f'<code>{ih}</code>'),
                ("Symbol",          symbol),
                ("Mode",            mode),
                ("Timestamp UTC",   ts),
                ("Accuracy Saved",  '<b style="color:#DC2626">NO</b>'),
            ]),
            unsafe_allow_html=True,
        )
    with c2:
        prov_rows = []
        for k, v in ps.items():
            color = "#DC2626" if v in ("FAILED", "NOT_AVAILABLE_ON_PLAN", "FAILED_DNS", "NOT_CONFIGURED", "AUTH_FAILED") \
                   else "#D97706" if v in ("PENDING", "NOT_RUN", "SKIPPED") \
                   else "#059669"
            prov_rows.append((k, f'<span style="color:{color};font-weight:700">{v}</span>'))
        if prov_rows:
            st.markdown(_table(prov_rows), unsafe_allow_html=True)

    _err_card(f"<b>Failure reason:</b> {error_msg}")

    if errs:
        for e in errs:
            _err_card(f"<b>Detail:</b> {e}")

    st.markdown(
        _table(_next_actions),
        unsafe_allow_html=True,
    )


def _section_header(n, title: str, sub: str = ""):
    sub_html = (
        f'<span style="color:#8BA9C4;font-size:.67rem;margin-left:.8rem">{sub}</span>'
        if sub else ""
    )
    num_html = (
        f'<span style="color:#60A5FA;font-weight:700;margin-right:.5rem">'
        f"Section {n} --</span>"
        if n else ""
    )
    st.markdown(
        f'<div style="background:#0A2540;padding:.5rem 1rem;border-radius:6px;'
        f'margin:1.1rem 0 .4rem 0">'
        f'<span style="color:#F1F5F9;font-weight:800;font-size:.9rem">'
        f"{num_html}{title}</span>{sub_html}</div>",
        unsafe_allow_html=True,
    )


def _table(rows):
    cells = "".join(
        f"<tr>"
        f'<td style="color:#425466;font-size:.73rem;font-weight:600;'
        f'padding:.28rem .5rem;width:44%;border-bottom:1px solid #F0F4F8">'
        f"{lbl}</td>"
        f'<td style="color:#0A2540;font-weight:700;font-size:.78rem;'
        f'padding:.28rem .5rem;border-bottom:1px solid #F0F4F8">'
        f"{val}</td></tr>"
        for lbl, val in rows
    )
    return f'<table style="width:100%;border-collapse:collapse">{cells}</table>'


def _table_dark(rows):
    """Table for dark-background panels — white/light text so it is always readable."""
    cells = "".join(
        f"<tr>"
        f'<td style="color:#CBD5E1;font-size:.73rem;font-weight:600;'
        f'padding:.28rem .5rem;width:44%;border-bottom:1px solid rgba(255,255,255,0.18)">'
        f"{lbl}</td>"
        f'<td style="color:#F8FAFC;font-weight:700;font-size:.78rem;'
        f'padding:.28rem .5rem;border-bottom:1px solid rgba(255,255,255,0.18)">'
        f"{val}</td></tr>"
        for lbl, val in rows
    )
    return f'<table style="width:100%;border-collapse:collapse">{cells}</table>'


def _err_color(v, suffix=""):
    if v is None:
        return "---"
    try:
        f = float(v)
        col = "#DF1B41" if abs(f) > 20 else ("#F59E0B" if abs(f) > 5 else "#00875A")
        return (
            f'<span style="color:{col};font-weight:700">'
            f"{'+'if f >= 0 else ''}{f:.2f}{suffix}</span>"
        )
    except Exception:
        return str(v)


def _render_decision_distribution_diagnostics():
    """Show decision distribution from past eval records. Warns if HOLD% is too high."""
    from pathlib import Path
    import json as _json
    eval_file = Path(__file__).resolve().parent / "stock_prediction_evaluation_runs.jsonl"
    records: list = []
    if eval_file.exists():
        try:
            with open(eval_file, "r", encoding="utf-8") as _fh:
                for _ln in _fh:
                    _ln = _ln.strip()
                    if _ln:
                        try:
                            records.append(_json.loads(_ln))
                        except Exception:
                            pass
        except Exception:
            pass

    total = len(records)
    if total == 0:
        _info_card("No past evaluation records yet — decision distribution will appear after first validated run.")
        return

    dec_counts: Dict[str, int] = {"BUY": 0, "SELL": 0, "HOLD": 0, "REVIEW": 0, "OTHER": 0}
    dec_ok = dir_ok = 0
    ret_errs: List[float] = []

    for r in records:
        ai_pred = r.get("ai_prediction", {}) or {}
        cmp_r   = r.get("comparison", {}) or {}
        dec = str(ai_pred.get("decision", "OTHER") or "OTHER").upper()
        dec_counts[dec if dec in dec_counts else "OTHER"] += 1
        if cmp_r.get("decision_match") is True:
            dec_ok += 1
        if cmp_r.get("directional_match") is True:
            dir_ok += 1
        if cmp_r.get("abs_return_error_pct") is not None:
            try:
                ret_errs.append(float(cmp_r["abs_return_error_pct"]))
            except Exception:
                pass

    hold_pct  = round(dec_counts["HOLD"] / total * 100, 1)
    dec_match = round(dec_ok / total * 100, 1) if total else 0
    dir_match = round(dir_ok / total * 100, 1) if total else 0
    avg_err   = round(sum(ret_errs) / len(ret_errs), 2) if ret_errs else None

    hold_color = "#EF4444" if hold_pct > 60 else ("#F59E0B" if hold_pct > 45 else "#10B981")
    dm_color   = "#10B981" if dec_match >= 60 else ("#F59E0B" if dec_match >= 45 else "#EF4444")
    dr_color   = "#10B981" if dir_match >= 55 else ("#F59E0B" if dir_match >= 40 else "#EF4444")
    err_color  = "#10B981" if (avg_err is not None and avg_err <= 10) else ("#F59E0B" if (avg_err is not None and avg_err <= 15) else "#EF4444")

    with st.expander("Decision Distribution Diagnostics (HOLD bias check)", expanded=False):
        st.markdown(
            _table([
                ("Total validated records",  str(total)),
                ("BUY decisions",            f'{dec_counts["BUY"]} ({round(dec_counts["BUY"]/total*100,1)}%)'),
                ("SELL decisions",           f'{dec_counts["SELL"]} ({round(dec_counts["SELL"]/total*100,1)}%)'),
                ("HOLD decisions",           f'<span style="color:{hold_color};font-weight:700">{dec_counts["HOLD"]} ({hold_pct}%)</span>'),
                ("REVIEW decisions",         f'{dec_counts["REVIEW"]} ({round(dec_counts["REVIEW"]/total*100,1)}%)'),
                ("Decision Match %",         f'<span style="color:{dm_color};font-weight:700">{dec_match}%</span>'),
                ("Direction Match %",        f'<span style="color:{dr_color};font-weight:700">{dir_match}%</span>'),
                ("Avg Return Error (pp)",    f'<span style="color:{err_color};font-weight:700">{avg_err if avg_err is not None else "---"}</span>'),
            ]),
            unsafe_allow_html=True,
        )
        if hold_pct > 60:
            st.error(f"HOLD% is {hold_pct}% — severely above 60% threshold. Signal score engine + Gemini prompt calibration needed.")
        elif hold_pct > 45:
            st.warning(f"HOLD% is {hold_pct}% — above 45% caution threshold. Model may be over-cautious.")
        else:
            st.success(f"HOLD% is {hold_pct}% — within acceptable range.")

        if dec_match < 50 or dir_match < 50 or (avg_err is not None and avg_err > 15):
            st.warning(
                f"CALIBRATION WARNING: Decision Match={dec_match}%, Direction Match={dir_match}%, "
                f"Avg Return Error={avg_err}pp. "
                "Model accuracy is below acceptable thresholds — predictions should be treated as experimental."
            )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════


def main():
    _apply_light_demo_theme()  # re-inject on every Streamlit rerun
    _render_batch_validation_sidebar()

    # ── TRUTH_GATE_V4 banner — visible in every mode, every run ──────────────
    import datetime as _dt_main
    _patch_now = _dt_main.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    _active_ai  = os.getenv("AI_PROVIDER", "gemini").upper()
    _gemini_key = bool(os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", ""))
    _allow_fb   = os.getenv("ALLOW_BASELINE_FALLBACK", "false")
    _tt_cred    = ("REFRESH" if os.getenv("TASTYTRADE_REFRESH_TOKEN")
                   else "ACCESS" if os.getenv("TASTYTRADE_ACCESS_TOKEN")
                   else "MISSING")
    _rapi_key   = bool(os.getenv("RAPIDAPI_KEY"))
    _banner_col = "#10B981" if _active_ai == "GEMINI" and _gemini_key else "#F59E0B"
    _tt_color   = "#EF4444" if _tt_cred == "MISSING" else ("#F59E0B" if _tt_cred == "ACCESS" else "#10B981")
    _rapi_color = "#EF4444" if not _rapi_key else "#10B981"
    st.markdown(
        f'<div style="background:#0D0D0D;border:2px solid #F59E0B;border-radius:6px;'
        f'padding:.45rem 1rem;margin-bottom:.4rem;font-size:.7rem">'
        f'<span style="color:#F59E0B;font-weight:900;font-size:.78rem">ACTIVE BUILD: INPUT_PREVIEW_OPTIONS_FIX_V1</span>&nbsp;&nbsp;'
        f'<span style="color:#9CA3AF">entrypoint: <b style="color:#D1D5DB">{_BUILD_FILE[-45:]}</b></span>&nbsp;&nbsp;'
        f'<span style="color:#9CA3AF">render: <b style="color:#D1D5DB">{_patch_now}</b></span>&nbsp;&nbsp;'
        f'<span style="color:#9CA3AF">AI: <b style="color:{_banner_col}">{_active_ai} {"KEY OK" if _gemini_key else "KEY MISSING"}</b></span>&nbsp;&nbsp;'
        f'<span style="color:#9CA3AF">TT cred: <b style="color:{_tt_color}">{_tt_cred}</b></span>&nbsp;&nbsp;'
        f'<span style="color:#9CA3AF">RapidAPI: <b style="color:{_rapi_color}">{"PRESENT" if _rapi_key else "MISSING"}</b></span>&nbsp;&nbsp;'
        f'<span style="color:#9CA3AF">Fallback: <b style="color:#D1D5DB">{_allow_fb.upper()}</b></span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    # FORCE_DISABLE_TASTYTRADE check — highest priority
    _force_disable_tt = os.getenv("FORCE_DISABLE_TASTYTRADE", "false").lower() == "true"
    if _force_disable_tt:
        st.error(
            "FORCE_DISABLE_TASTYTRADE=true — Tastytrade is DISABLED for verification. "
            "Options backtest will be blocked regardless of credentials. "
            "Remove this env var to re-enable."
        )
    if _tt_cred == "MISSING":
        st.warning(
            "TASTYTRADE CREDENTIALS MISSING — No TASTYTRADE_REFRESH_TOKEN or TASTYTRADE_ACCESS_TOKEN in .env. "
            "Options backtest will be blocked. Set refresh token for persistent auth."
        )
    if _tt_cred == "ACCESS":
        # Try to detect staleness from JWT exp claim (non-sensitive — just the exp timestamp)
        _tt_stale_msg = None
        try:
            import base64 as _b64, json as _json_jwt
            _raw_tok = os.getenv("TASTYTRADE_ACCESS_TOKEN", "")
            _parts = _raw_tok.split(".")
            if len(_parts) == 3:
                _padded = _parts[1] + "=" * (-len(_parts[1]) % 4)
                _payload = _json_jwt.loads(_b64.urlsafe_b64decode(_padded))
                _exp_ts  = _payload.get("exp")
                if _exp_ts:
                    import datetime as _dt_jwt
                    _exp_dt  = _dt_jwt.datetime.utcfromtimestamp(_exp_ts)
                    _now_jwt = _dt_jwt.datetime.utcnow()
                    if _exp_dt < _now_jwt:
                        _tt_stale_msg = (
                            f"TASTYTRADE ACCESS TOKEN IS EXPIRED (expired at {_exp_dt.strftime('%Y-%m-%d %H:%M')} UTC, "
                            f"now {_now_jwt.strftime('%Y-%m-%d %H:%M')} UTC). "
                            "Options backtest will fail. Paste a fresh access token or add a refresh token."
                        )
                    else:
                        _mins_left = int((_exp_dt - _now_jwt).total_seconds() / 60)
                        if _mins_left < 5:
                            _tt_stale_msg = (
                                f"TASTYTRADE ACCESS TOKEN EXPIRES IN {_mins_left} MIN ({_exp_dt.strftime('%H:%M')} UTC). "
                                "Refresh now or options backtest will fail mid-run."
                            )
        except Exception:
            pass
        if _tt_stale_msg:
            st.error(_tt_stale_msg)
        else:
            st.warning(
                "TASTYTRADE ACCESS TOKEN ONLY (no refresh token) — Access tokens expire every ~15 minutes. "
                "App will fail for options if token is stale. Add TASTYTRADE_REFRESH_TOKEN for persistent auth."
            )
    # RapidAPI capability panel (honest about what current plan supports)
    if _rapi_key:
        with st.expander("RapidAPI Plan Capabilities (click to verify what is available)"):
            st.markdown(
                '<div style="font-size:.74rem;line-height:1.7">'
                '<b style="color:#FCD34D">RapidAPI key is present.</b> '
                'What each endpoint actually provides on the current plan:<br>'
                '<table style="width:100%;font-size:.71rem;border-collapse:collapse">'
                '<tr style="color:#93C5FD"><th style="text-align:left;padding:.15rem .4rem">Endpoint</th>'
                '<th style="text-align:left;padding:.15rem .4rem">Used for</th>'
                '<th style="text-align:left;padding:.15rem .4rem">Status</th></tr>'
                '<tr><td style="padding:.12rem .4rem">TradingView quote (OHLCV)</td>'
                '<td style="padding:.12rem .4rem">Historical OHLCV bars</td>'
                '<td style="color:#F59E0B;padding:.12rem .4rem">PLAN-DEPENDENT — may not be on current plan; external_historical_provider used as fallback</td></tr>'
                '<tr><td style="padding:.12rem .4rem">Market movers / screener (/market/get-movers)</td>'
                '<td style="padding:.12rem .4rem">Live market movers — sent to Gemini for LIVE predictions (origin within 7 days); excluded for historical runs (leakage prevention)</td>'
                '<td style="color:#10B981;padding:.12rem .4rem">ACTIVE — wired into Gemini prompt</td></tr>'
                '<tr><td style="padding:.12rem .4rem">Options chain data</td>'
                '<td style="padding:.12rem .4rem">Live strike/IV for exact-strike selection</td>'
                '<td style="color:#EF4444;padding:.12rem .4rem">NOT WIRED — exact-strike unsupported by TT backtest</td></tr>'
                '<tr><td style="padding:.12rem .4rem">Earnings calendar (/api/calendar/earnings)</td>'
                '<td style="padding:.12rem .4rem">Earnings warning in AI prompt for prediction window</td>'
                '<td style="color:#10B981;padding:.12rem .4rem">ACTIVE — checked on every run; warning injected if earnings found</td></tr>'
                '</table>'
                '<br><span style="color:#9CA3AF">To use additional endpoints: wire them into the Gemini feature packet '
                '(see calibration_profile_builder.py and failure_analyzer.py for roadmap).</span>'
                '</div>',
                unsafe_allow_html=True,
            )
    else:
        with st.expander("RapidAPI Key Missing — Impact on This Run"):
            st.markdown(
                '<div style="font-size:.74rem;color:#FCA5A5">'
                '<b>RAPIDAPI_KEY not set in .env</b><br>'
                'Historical data falls back to yfinance (free, rate-limited, no options chain).<br>'
                'No market-mover signals available. Add RAPIDAPI_KEY to unlock TradingView endpoint.'
                '</div>',
                unsafe_allow_html=True,
            )

    # Page header
    st.markdown(
        '<div style="background:#0A2540;padding:1rem 1.5rem;border-radius:8px;'
        'margin-bottom:.6rem">'
        '<div style="font-size:1.35rem;font-weight:900;color:#F1F5F9;'
        'letter-spacing:-.02em">'
        "AI Financial Analyst System — Walk-Forward Prediction Validation"
        "</div>"
        '<div style="font-size:.72rem;color:#8BA9C4;margin-top:.3rem">'
        "AI predicts from the prediction origin date to the target date. "
        "Actual validation/backtesting checks what really happened in that same window. "
        "Two-window walk-forward architecture — no data leakage."
        "</div></div>",
        unsafe_allow_html=True,
    )

    # Mode selector
    mode = st.radio(
        "Select Validation Mode",
        ["Stock Price Validation", "Options Strategy Validation"],
        horizontal=True,
        help=(
            "Stock Price Validation: AI predicts stock price movement vs actual historical close prices.\n"
            "Options Strategy Validation: AI predicts + tastytrade options backtest "
            "(backtest runs ONLY for prediction window -- never for historical context window)."
        ),
    )

    if mode == "Stock Price Validation":
        _info_card(
            "<b>Mode 1 -- Stock Price Validation:</b> "
            "Select a stock and a past prediction origin date. "
            "AI predicts price movement in the prediction window. "
            "Actual validation checks real historical prices in that same window. "
            "No options fields. No strategy parameters. Honest price-to-price comparison."
        )
        _render_stock_form()

        if st.session_state.get("ai_result") and st.session_state.get("active_mode") == "stock":
            _render_results()
        if st.session_state.get("error_msg") and not st.session_state.get("ai_result"):
            _render_failure_panel()

    else:
        _info_card(
            "<b>Mode 2 -- Options Strategy Validation:</b> "
            "AI uses the historical context window for pattern analysis. "
            "The options backtest runs from <b>prediction origin → target date</b> "
            "(same window as the AI prediction). "
            "Validate whether the options strategy P&amp;L matches the AI's directional call."
        )
        if not _TT_AVAILABLE:
            _warn_card(
                "<b>Tastytrade Backtester Unavailable:</b> "
                "Could not import tastytrade backtester. "
                "AI prediction will still run. "
                "Check src/services/ and required dependencies."
            )
        _render_options_form()

        if st.session_state.get("ai_result") and st.session_state.get("active_mode") == "options":
            _render_options_results()
        if st.session_state.get("error_msg") and not st.session_state.get("ai_result"):
            _render_failure_panel()

    # Section 6 -- Accuracy log always visible
    _render_records()

    # Sidebar rolling validation
    _sidebar_rolling()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 -- STOCK FORM
# ══════════════════════════════════════════════════════════════════════════════


def _render_stock_form():
    _section_header(1, "Prediction Setup", "Stock + dates + capital")

    with st.form("stock_form", clear_on_submit=False):
        # Dynamic defaults — always relative to today
        import datetime as _dt_sf
        _sf_today   = _dt_sf.date.today()
        _sf_origin  = (_sf_today - _dt_sf.timedelta(days=60)).isoformat()
        _sf_ctx     = (_sf_today - _dt_sf.timedelta(days=425)).isoformat()

        col1, col2, col3 = st.columns(3)
        with col1:
            symbol = st.text_input(
                "Stock Symbol", value="SPY",
                help="Use SPY or SPX for most reliable 1-month predictions. TSLA/NVDA also supported.",
            )
        with col2:
            benchmark = st.text_input(
                "Benchmark", value="SPY",
                help="Market benchmark e.g. SPY, QQQ, DIA, IWM",
            )
        with col3:
            capital = st.number_input(
                "Initial Capital ($)", value=50_000, min_value=100, step=1_000, format="%d",
            )

        _render_symbol_accuracy_warning(symbol)

        col4, col5, col6 = st.columns(3)
        with col4:
            ctx_start = st.text_input(
                "Historical Context Start Date", value=_sf_ctx,
                help="AI uses price history FROM this date. Auto-set to 14 months before today. (YYYY-MM-DD)",
            )
        with col5:
            origin_dt = st.text_input(
                "Prediction Origin Date", value=_sf_origin,
                help="AI predicts AS OF this date. Auto-set to 60 days ago. (YYYY-MM-DD)",
            )
        with col6:
            horizon = st.number_input(
                "Decision Horizon (days)", value=30, min_value=1, max_value=365,
                help="Days forward from origin date to target date.",
            )

        val_mode    = "horizon_days"
        price_basis = st.selectbox(
            "Price Basis", options=["close", "open"], index=0,
            help="close = standard for walk-forward validation (end-of-day price). open = next-day open.",
            key="price_basis_stock",
        )

        try:
            tgt_dt = datetime.strptime(origin_dt.strip(), "%Y-%m-%d") + timedelta(days=int(horizon))
            tgt_str = tgt_dt.strftime("%Y-%m-%d")
        except Exception:
            tgt_str = "---"

        _window_preview(ctx_start, origin_dt, tgt_str, horizon)

        require_strict_coverage = False

        submitted = st.form_submit_button(
            "Run Walk-Forward Prediction and Validation",
            type="primary", use_container_width=True,
        )

    if submitted:
        _clear_run()
        run_id = str(uuid.uuid4())[:8].upper()
        st.session_state["run_id"] = run_id
        st.session_state["active_mode"] = "stock"
        # Capture exact user-entered inputs immediately after submit
        st.session_state["user_input_snapshot"] = {
            "symbol":                        symbol.strip().upper(),
            "benchmark":                     benchmark.strip().upper(),
            "initial_capital":               float(capital),
            "historical_context_start_date": ctx_start.strip(),
            "prediction_origin_date":        origin_dt.strip(),
            "decision_horizon_days":         int(horizon),
            "target_date":                   tgt_str,
            "validation_mode":               val_mode,
            "price_basis":                   price_basis,
            "strict_coverage_required":      require_strict_coverage,
            "provider_constrained_allowed":  not require_strict_coverage,
            "run_id":                        run_id,
            "submitted_at_utc":              datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        _run_all(
            symbol=symbol.strip().upper(),
            ctx_start=ctx_start.strip(),
            origin_date=origin_dt.strip(),
            horizon=int(horizon),
            capital=float(capital),
            benchmark=benchmark.strip().upper(),
            val_mode=val_mode,
            target_date=tgt_str,
            run_id=run_id,
            price_basis=price_basis,
            allow_provider_truncation=not require_strict_coverage,
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 -- OPTIONS FORM
# ══════════════════════════════════════════════════════════════════════════════


def _render_options_form():
    _section_header(
        1, "Options Strategy Validation Setup",
        "Stock + dates + capital + tastytrade-style options parameters",
    )

    # Delta is the only strike selection mode used by Tastytrade backtester
    strike_selection = "Delta"
    st.session_state["opt_strike_sel"] = "Delta"

    # ── Exit Conditions OUTSIDE form — toggles update UI immediately ──────────
    # Placing these outside st.form() lets checkbox changes instantly show/hide
    # the value inputs, the same way TastyTrade's toggle switches work.
    st.markdown(
        '<div style="background:#0F2940;border:1px solid #1E4D7A;border-radius:6px;'
        'padding:.5rem 1rem;margin:.5rem 0 .4rem 0">'
        '<span style="color:#93C5FD;font-weight:800;font-size:.82rem">Exit Conditions</span>'
        '<span style="color:#64748B;font-size:.7rem;margin-left:.8rem">'
        'Toggle each condition on. Value only applies when enabled.</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    _ec1, _ec2, _ec3 = st.columns(3)

    with _ec1:
        _tp_on = st.checkbox(
            "Take Profit at % of Premium",
            value=False, key="opt_tp_on",
        )
        if _tp_on:
            take_profit_pct_val = st.number_input(
                "Take Profit %", value=50, min_value=1, max_value=1000, step=5,
                key="opt_tp_pct",
                help="Exit when profit reaches this % of initial premium. e.g. 50 = exit at 50% gain.",
            )
        else:
            take_profit_pct_val = 0

    with _ec2:
        _sl_on = st.checkbox(
            "Stop Loss at % of Premium",
            value=False, key="opt_sl_on",
        )
        if _sl_on:
            stop_loss_pct_val = st.number_input(
                "Stop Loss %", value=20, min_value=1, max_value=1000, step=5,
                key="opt_sl_pct",
                help="Exit when loss reaches this % of initial premium. e.g. 20 = exit at 20% loss.",
            )
        else:
            stop_loss_pct_val = 0

    with _ec3:
        _ed_on = st.checkbox(
            "Exit After N Days in Trade",
            value=False, key="opt_ed_on",
        )
        if _ed_on:
            exit_after_days_val = st.number_input(
                "Days", value=2, min_value=1, max_value=365, step=1,
                key="opt_ed_days",
                help="Exit the trade after this many calendar days, regardless of P&L.",
            )
        else:
            exit_after_days_val = 0

    with st.form("options_form", clear_on_submit=False):
        # ── Dynamic defaults: always relative to today so they never go stale ──
        import datetime as _dt_defaults
        _today = _dt_defaults.date.today()
        _default_origin  = (_today - _dt_defaults.timedelta(days=90)).isoformat()   # 90 days ago — safely in TastyTrade DB
        _default_ctx     = (_today - _dt_defaults.timedelta(days=455)).isoformat()  # ~15 months back
        _default_horizon = 30

        # ── Base inputs (same as stock mode) ─────────────────────────────────
        col1, col2, col3 = st.columns(3)
        with col1:
            symbol = st.text_input(
                "Stock Symbol", value="SPY",
                help="SPY or SPX recommended — lower randomness than single stocks. TSLA also works.",
            )
        with col2:
            benchmark = st.text_input("Benchmark", value="SPY")
        with col3:
            capital = st.number_input(
                "Initial Capital ($)", value=50_000, min_value=100, step=1_000, format="%d",
            )

        _render_symbol_accuracy_warning(symbol)

        col4, col5, col6 = st.columns(3)
        with col4:
            ctx_start = st.text_input(
                "Historical Context Start Date", value=_default_ctx,
                help="AI context window start only. Options backtest starts at Prediction Origin Date. Auto-set to 14 months before today.",
            )
        with col5:
            origin_dt = st.text_input(
                "Prediction Origin Date", value=_default_origin,
                help="Options backtest STARTS here (origin → target). AI predicts from this date forward. Auto-set to 60 days ago.",
            )
        with col6:
            horizon = st.number_input(
                "Decision Horizon (days)", value=_default_horizon, min_value=1, max_value=365,
                help="Options backtest ENDS at origin + N days.",
            )

        val_mode    = "horizon_days"
        price_basis = st.selectbox(
            "Price Basis", options=["close", "open"], index=0,
            help="close = standard for walk-forward validation. open = next-day open.",
            key="price_basis_options",
        )

        try:
            tgt_dt = datetime.strptime(origin_dt.strip(), "%Y-%m-%d") + timedelta(days=int(horizon))
            tgt_str = tgt_dt.strftime("%Y-%m-%d")
        except Exception:
            tgt_str = "---"

        _window_preview(ctx_start, origin_dt, tgt_str, horizon)

        # ── Options Strategy Parameters ───────────────────────────────────────
        st.markdown(
            '<div style="background:#0F2940;border:1px solid #1E4D7A;border-radius:6px;'
            'padding:.6rem 1rem;margin:.6rem 0">'
            '<span style="color:#93C5FD;font-weight:800;font-size:.85rem">'
            "Options Strategy Parameters</span>"
            '<span style="color:#64748B;font-size:.7rem;margin-left:.8rem">'
            "Backtest runs from prediction origin date to target date"
            "</span></div>",
            unsafe_allow_html=True,
        )

        oc1, oc2, oc3 = st.columns(3)
        with oc1:
            direction = st.selectbox(
                "Direction",
                options=["Buy", "Sell"],
                help="Buy = long position. Sell = short position (write/sell an option).",
            )
        with oc2:
            opt_type = st.selectbox(
                "Type",
                options=["Call", "Put"],
                help="Call option or Put option.",
            )
        with oc3:
            quantity = st.number_input(
                "Quantity (contracts)",
                value=1, min_value=1, max_value=10, step=1,
                help="Number of option contracts. Tastytrade API limit: 1–10.",
            )

        # ── Entry Schedule row ────────────────────────────────────────────────
        es_col, _ = st.columns([1, 2])
        with es_col:
            entry_schedule = st.selectbox(
                "Entry Schedule",
                options=["Every day", "On exact DTE match", "Weekly", "Monthly"],
                index=0,
                key="opt_entry_schedule",
                help=(
                    "Every day — enter a new trade on every available trading day. "
                    "On exact DTE match — only enter when a contract with exactly the chosen DTE exists. "
                    "Weekly — enter once per week. Monthly — enter once per month."
                ),
            )
        # strike_selection is defined OUTSIDE this form (above) so it triggers re-render on change

        # ── Conditional inputs based on strike selection ──────────────────
        if strike_selection == "Delta":
            od1, od2 = st.columns(2)
            with od1:
                delta_val = st.number_input(
                    "Delta (1-99)",
                    value=30, min_value=1, max_value=99,
                    help="Target delta for strike selection (e.g. 46 = 0.46 delta, near ATM).",
                )
            with od2:
                dte_val = st.number_input(
                    "Expiration (DTE)",
                    value=30, min_value=1, max_value=365,
                    help="Days to expiration at entry. Use 30 to match the 1-month prediction horizon.",
                )
            # Delta education panel
            _delta_disp_e = int(delta_val)
            _delta_itm_prob = _delta_disp_e
            _delta_prem_move = _delta_disp_e / 100.0
            _delta_zone = (
                "Deep In-The-Money (ITM)" if _delta_disp_e >= 70
                else "At-The-Money (ATM)" if _delta_disp_e >= 45
                else "Near Out-Of-The-Money" if _delta_disp_e >= 25
                else "Far Out-Of-The-Money (OTM)"
            )
            _zone_color = (
                "#10B981" if _delta_disp_e >= 70
                else "#FCD34D" if _delta_disp_e >= 45
                else "#F97316" if _delta_disp_e >= 25
                else "#EF4444"
            )
            st.markdown(
                f'<div style="background:#0F172A;border-left:3px solid {_zone_color};'
                f'border-radius:4px;padding:.45rem .8rem;margin:.3rem 0;font-size:.75rem;color:#CBD5E1">'
                f'<span style="color:{_zone_color};font-weight:700">Delta {_delta_disp_e} — {_delta_zone}</span>'
                f'&nbsp; | &nbsp;'
                f'<span style="color:#94A3B8">~{_delta_itm_prob}% probability of expiring in-the-money</span>'
                f'&nbsp; | &nbsp;'
                f'<span style="color:#94A3B8">Premium moves ~<b>${_delta_prem_move:.2f}</b> for every $1 stock move</span>'
                f'&nbsp; | &nbsp;'
                f'<span style="color:#94A3B8">Recommended for 1-month horizon: <b>Delta 30-50</b></span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if int(dte_val) < 7:
                st.warning(
                    f"**DTE {int(dte_val)} is very short** — options expiring in {int(dte_val)} day(s) have "
                    f"extreme time decay and very high sensitivity to daily price moves. "
                    f"**Recommended for 1-month backtesting: DTE 14–45.** "
                    f"Set DTE to 30 to match the 30-day prediction horizon."
                )
            otm_pct_val     = 5.0
            price_offset_val = 5.0
            premium_target_val = 1.0
            strike_price    = 0.0
            expiry_date_str = ""

        allow_proxy_for_exact = False
        exit_rule        = "combined"
        otm_pct_val      = 5.0
        price_offset_val = 5.0
        premium_target_val = 1.0

        _params_summary = f"{direction} {opt_type}, Delta {delta_val}, DTE {dte_val}, Qty {quantity}"

        st.markdown(
            f'<div style="background:#F0FFF4;border:1px solid #6EE7B7;border-radius:6px;'
            f'padding:.5rem 1rem;font-size:.74rem;color:#065F46;margin:.3rem 0">'
            f'<b>Options Backtest Window:</b>&emsp;'
            f'<code>{origin_dt}</code> to <code>{tgt_str}</code>&emsp;'
            f'&mdash;&emsp;{_params_summary}<br>'
            f'<span style="color:#059669">'
            f'Backtest runs from prediction origin ({origin_dt}) to target ({tgt_str}) — same window as AI prediction.'
            f'</span></div>',
            unsafe_allow_html=True,
        )
        submitted = st.form_submit_button(
            "Run AI Prediction + Options Strategy Backtest",
            type="primary", use_container_width=True,
        )

    if submitted:
        _clear_run()
        run_id = str(uuid.uuid4())[:8].upper()
        st.session_state["run_id"] = run_id
        st.session_state["active_mode"] = "options"

        _is_strike_mode     = False   # Exact-strike mode removed
        _is_delta_mode      = (strike_selection == "Delta")
        _is_delta_mode_snap = _is_delta_mode
        # Capture exact user-entered options inputs immediately after submit
        st.session_state["user_input_snapshot"] = {
            "symbol":                        symbol.strip().upper(),
            "benchmark":                     benchmark.strip().upper(),
            "initial_capital":               float(capital),
            "historical_context_start_date": ctx_start.strip(),
            "prediction_origin_date":        origin_dt.strip(),
            "decision_horizon_days":         int(horizon),
            "target_date":                   tgt_str,
            "validation_mode":               val_mode,
            "price_basis":                   price_basis,
            "strike_selection":              strike_selection,
            "direction":                     direction,
            "opt_type":                      opt_type,
            "quantity":                      int(quantity),
            "delta_ui":                      int(delta_val) if _is_delta_mode_snap else None,
            "otm_pct":                       float(otm_pct_val) if strike_selection == "Percentage OTM" else None,
            "price_offset":                  float(price_offset_val) if strike_selection == "Price Offset From Underlying" else None,
            "premium_target":                float(premium_target_val) if strike_selection == "Premium" else None,
            "strike_price":                  None,
            "expiry_date":                   None,
            "dte":                           int(dte_val),
            "entry_schedule":                entry_schedule,
            "take_profit_pct":               float(take_profit_pct_val) if take_profit_pct_val > 0 else None,
            "stop_loss_pct":                 float(stop_loss_pct_val)   if stop_loss_pct_val > 0   else None,
            "exit_after_days":               int(exit_after_days_val)   if exit_after_days_val > 0 else None,
            "run_id":                        run_id,
            "submitted_at_utc":              datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        _req_strike   = None
        _expiry_clean = None

        # Entry frequency: map UI label → Tastytrade API value (all confirmed working)
        _entry_freq_map = {
            "Every day":           "every day",
            "On exact DTE match":  "on_exact_dte_match",
            "Weekly":              "weekly",
            "Monthly":             "monthly",
        }
        _api_entry_freq = _entry_freq_map.get(entry_schedule, "every day")

        # ── Single source of truth for all options inputs ─────────────────────
        options_params = {
            "direction":        direction,
            "opt_type":         opt_type,
            "quantity":         int(quantity),
            "strike_selection": strike_selection.lower(),
            "contract_selection_method": "DELTA_SELECTION" if _is_delta_mode else strike_selection.upper().replace(" ", "_"),
            # Delta mode fields
            "delta_ui":         int(delta_val) if _is_delta_mode else None,
            "delta_decimal":    round(int(delta_val) / 100.0, 4) if _is_delta_mode else None,
            "delta":            int(delta_val) if _is_delta_mode else None,
            # New strike selection fields
            "otm_pct":          float(otm_pct_val) if strike_selection == "Percentage OTM" else None,
            "price_offset":     float(price_offset_val) if strike_selection == "Price Offset From Underlying" else None,
            "premium_target":   float(premium_target_val) if strike_selection == "Premium" else None,
            # Strike fields — always None (exact-strike mode removed)
            "requested_strike": None,
            "strike_price":     None,
            "expiry_date":      None,
            # DTE valid in all modes
            "dte":              int(dte_val),
            "entry_schedule":     entry_schedule,
            "api_entry_frequency": _api_entry_freq,
            "exit_rule":          exit_rule,
            "take_profit_pct":    float(take_profit_pct_val) if take_profit_pct_val > 0 else None,
            "stop_loss_pct":      float(stop_loss_pct_val)   if stop_loss_pct_val > 0   else None,
            "exit_after_days":    int(exit_after_days_val)   if exit_after_days_val > 0 else None,
            "allow_proxy_for_exact": False,
        }

        if True:  # no pre-validation needed (strike mode removed)
            _run_options_all(
                symbol=symbol.strip().upper(),
                ctx_start=ctx_start.strip(),
                origin_date=origin_dt.strip(),
                horizon=int(horizon),
                capital=float(capital),
                benchmark=benchmark.strip().upper(),
                val_mode=val_mode,
                target_date=tgt_str,
                run_id=run_id,
                price_basis=price_basis,
                options_params=options_params,
            )


def _window_preview(ctx_start, origin_dt, tgt_str, horizon):
    st.markdown(
        f'<div style="background:#F0F9FF;border:1px solid #BAE6FD;'
        f'border-radius:6px;padding:.55rem 1.1rem;font-size:.74rem;'
        f'color:#0C4A6E;margin:.3rem 0">'
        f"<b>Historical Context Window</b>&emsp;"
        f"<code>{ctx_start}</code> to <code>{origin_dt}</code>"
        f"&emsp;&emsp;|&emsp;&emsp;"
        f"<b>Prediction / Validation Window</b>&emsp;"
        f"<code>{origin_dt}</code> to <code>{tgt_str}</code> ({horizon} days)"
        f'<br><span style="color:#0C4A6E;margin-top:.3rem;display:block">'
        f"AI uses context window for learning. "
        f"AI predicts the validation window. Actual data validates that same window."
        f"</span></div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATORS
# ══════════════════════════════════════════════════════════════════════════════


def _run_all(
    symbol, ctx_start, origin_date, horizon, capital, benchmark,
    val_mode, target_date, run_id, price_basis="close",
    allow_provider_truncation=False,
):
    """Stock Price Validation orchestrator."""
    today = date.today()

    spi = {
        "symbol":                        symbol,
        "historical_context_start_date": ctx_start,
        "prediction_origin_date":        origin_date,
        "decision_horizon_days":         horizon,
        "target_date":                   target_date,
        "initial_capital":               capital,
        "benchmark":                     benchmark,
        "validation_mode":               val_mode,
        "price_basis":                   price_basis,
    }

    # Detect run type: live future prediction vs historical validation
    try:
        _target_dt = datetime.strptime(target_date, "%Y-%m-%d").date()
    except Exception:
        _target_dt = today
    run_type = "live_future_prediction" if _target_dt > today else "historical_validation"

    valid, err = validate_stock_prediction_input(spi)
    if not valid:
        st.session_state["error_msg"] = f"Input validation failed: {err}"
        return

    input_hash = build_stock_prediction_hash(spi)
    _run_ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    _run_ctx = {
        "run_id":     run_id,
        "input_hash": input_hash,
        "mode":       "stock",
        "symbol":     symbol,
        "timestamp_utc": _run_ts,
        "provider_status": {
            "historical_data": "PENDING",
            "rapidapi":        "PENDING",
            "tastytrade":      "NOT_APPLICABLE",
            "gemini":          "PENDING",
        },
        "validation_status": "STARTED",
        "errors": [],
        "accuracy_eligible": False,
    }
    st.session_state.update({
        "spi":         spi,
        "input_hash":  input_hash,
        "run_type":    run_type,
        "run_ts":      _run_ts,
        "run_context": _run_ctx,
    })

    _today_str = today.strftime("%Y-%m-%d")

    with st.spinner(f"Fetching historical prices for {symbol} (trying multiple providers)..."):
        hist, hist_err, _hist_cov = fetch_price_history_for_range(
            symbol, ctx_start, _today_str
        )

    # Propagate coverage info to run_context early so later checks can read it
    _run_ctx["provider_chain"] = _hist_cov.get("providers_tried", [])
    _run_ctx["hist_provider"]  = _hist_cov.get("provider", "unknown")
    _run_ctx["effective_ctx_start"]  = _hist_cov.get("effective_start") or ctx_start
    _run_ctx["requested_ctx_start"]  = ctx_start
    _run_ctx["ctx_coverage_status"]  = _hist_cov.get("coverage_status", "FAILED")

    if not hist:
        _run_ctx["provider_status"]["historical_data"] = "FAILED"
        _run_ctx["provider_status"]["gemini"]          = "NOT_RUN"
        _run_ctx["validation_status"] = "HISTORICAL_DATA_UNAVAILABLE"
        _run_ctx["errors"].append(hist_err or "Historical price data unavailable")
        st.session_state["run_context"] = _run_ctx
        st.session_state["error_msg"]   = hist_err
        _log_invalid_run(run_id, input_hash, symbol, "stock",
                         "HISTORICAL_DATA_UNAVAILABLE", hist_err or "")
        return

    _run_ctx["provider_status"]["historical_data"] = "SUCCESS"
    st.session_state["price_hist"]  = hist
    st.session_state["ctx_summary"] = get_context_summary(hist, ctx_start, origin_date)

    # ── Paid API health checks (proof of live API calls) ─────────────────────
    _force_disable_tt_stock = os.getenv("FORCE_DISABLE_TASTYTRADE", "false").lower() == "true"
    with st.spinner("Calling paid APIs (RapidAPI health check)..."):
        _rapi_result = _rapidapi_hc()
        if _force_disable_tt_stock:
            _tt_result = {
                "called": False, "endpoint": "NOT_CALLED",
                "http_status": None, "latency_ms": None,
                "token_refreshed": False, "customer_verified": False,
                "refresh_present": False, "role": "DISABLED_BY_FORCE_DISABLE_TASTYTRADE",
                "used_in_stock_mode": False, "used_in_options_mode": False,
                "error": "FORCE_DISABLE_TASTYTRADE=true — not called",
                "blocked_by": "MISSING_FORCED", "backtest_allowed": False,
            }
        else:
            _tt_result = _tastytrade_hc()
    st.session_state["rapidapi_health"]   = _rapi_result
    st.session_state["tastytrade_health"] = _tt_result
    _run_ctx["provider_status"]["rapidapi"] = "CALLED" if _rapi_result.get("called") else "NOT_CONFIGURED"
    _run_ctx["provider_status"]["tastytrade"] = "FORCE_DISABLED" if _force_disable_tt_stock else (
        "VERIFIED" if _tt_result.get("customer_verified") else "NOT_USED_IN_STOCK_MODE"
    )
    st.session_state["run_context"] = _run_ctx

    # Strict two-bound context filter: AI sees ONLY [ctx_start, origin_date]
    ctx_bars = [b for b in hist if ctx_start <= b["date"] <= origin_date]
    if not ctx_bars:
        _run_ctx["provider_status"]["gemini"] = "NOT_RUN"
        _run_ctx["errors"].append(
            f"No price bars between {ctx_start} and {origin_date}. "
            "Provider returned bars outside this range or no data for these dates."
        )
        st.session_state["run_context"] = _run_ctx
        st.session_state["error_msg"] = (
            f"No price bars found between {ctx_start} and {origin_date}. "
            f"Provider returned {len(hist)} bars but none in your requested range. "
            f"Earliest available: {hist[0]['date'] if hist else 'N/A'}. "
            f"Latest available: {hist[-1]['date'] if hist else 'N/A'}. "
            "If you entered a date like 2017, the provider may not have data that far back."
        )
        _log_invalid_run(run_id, input_hash, symbol, "stock",
                         "NO_BARS_IN_REQUESTED_DATE_RANGE",
                         f"Requested {ctx_start} to {origin_date}, provider has {hist[0]['date'] if hist else 'N/A'} to {hist[-1]['date'] if hist else 'N/A'}")
        return

    # Pre-flight context validation
    _first_bar = ctx_bars[0]["date"]
    _last_bar  = ctx_bars[-1]["date"]

    # Effective date discrepancy — use coverage info from provider chain
    _effective_ctx_start = _first_bar
    _date_gap_days = 0
    _provider_name = _hist_cov.get("provider", "unknown")
    _cov_status    = _hist_cov.get("coverage_status", "PARTIAL")

    if _effective_ctx_start > ctx_start:
        try:
            _req_dt  = datetime.strptime(ctx_start, "%Y-%m-%d").date()
            _eff_dt  = datetime.strptime(_effective_ctx_start, "%Y-%m-%d").date()
            _date_gap_days = (_eff_dt - _req_dt).days
        except Exception:
            _date_gap_days = 0
        _run_ctx["effective_ctx_start"]    = _effective_ctx_start
        _run_ctx["requested_ctx_start"]    = ctx_start
        _run_ctx["ctx_start_gap_days"]     = _date_gap_days
        _run_ctx["provider_coverage"] = {
            "requested_ctx_start":        ctx_start,
            "first_available_bar":        _effective_ctx_start,
            "coverage_gap_days":          _date_gap_days,
            "coverage_status":            "TRUNCATED",
            "provider_used":              _provider_name,
            "providers_tried":            _hist_cov.get("providers_tried", []),
            "strict_mode":                not allow_provider_truncation,
            "provider_truncation_allowed": allow_provider_truncation,
            "run_allowed":                allow_provider_truncation,
        }
        if not allow_provider_truncation:
            # STRICT MODE: all providers tried — still couldn't reach requested date
            _providers_tried_str = ", ".join(
                p["provider"] for p in _hist_cov.get("providers_tried", [])
            ) or "none"
            _block_msg = (
                f"RUN BLOCKED — EXACT DATE RANGE NOT AVAILABLE\n\n"
                f"Requested Context Start : {ctx_start}\n"
                f"Provider First Available: {_effective_ctx_start} (gap: {_date_gap_days} days)\n"
                f"Providers tried         : {_providers_tried_str}\n\n"
                f"AI Prediction : NOT_RUN\n"
                f"Validation    : NOT_RUN\n"
                f"Accuracy Saved: NO\n\n"
                f"Reason: No provider (NASDAQ, yfinance) can supply data from {ctx_start}. "
                f"Earliest available: {_effective_ctx_start}. "
                f"The run was blocked because no provider covers the requested start date. "
                f"Try a more recent context start date (e.g. {_effective_ctx_start})."
            )
            _run_ctx["input_binding_warning"] = _block_msg
            _run_ctx["validation_status"] = "BLOCKED_DATE_RANGE_UNAVAILABLE"
            _run_ctx["provider_status"]["gemini"] = "NOT_RUN"
            st.session_state["run_context"] = _run_ctx
            st.session_state["error_msg"] = _block_msg
            _log_invalid_run(run_id, input_hash, symbol, "stock",
                             "BLOCKED_DATE_RANGE_UNAVAILABLE",
                             f"User requested {ctx_start}, best provider starts {_effective_ctx_start}, "
                             f"gap={_date_gap_days}d. Tried: {_providers_tried_str}.")
            return
        else:
            # PROVIDER-CONSTRAINED MODE (checkbox unchecked): warn but allow
            _run_ctx["input_binding_warning"] = (
                f"PROVIDER-CONSTRAINED RUN — requested context start was {ctx_start}, but "
                f"best available provider ({_provider_name}) first bar is {_effective_ctx_start} "
                f"(gap: {_date_gap_days} days). "
                f"AI used {_effective_ctx_start} as effective context start — NOT your requested {ctx_start}. "
                f"AI output is labeled PROVIDER-CONSTRAINED to reflect this."
            )
            _run_ctx["coverage_status"] = "PROVIDER_CONSTRAINED"
            _run_ctx["provider_coverage"]["coverage_status"] = "PROVIDER_CONSTRAINED"
            st.session_state["run_context"] = _run_ctx

    if _first_bar < ctx_start:
        st.session_state["error_msg"] = (
            f"AI context leakage detected: first bar {_first_bar} is before "
            f"selected context start {ctx_start}. Blocked."
        )
        return
    if _last_bar > origin_date:
        st.session_state["error_msg"] = (
            f"AI future leakage detected: last bar {_last_bar} is after "
            f"prediction origin {origin_date}. Blocked."
        )
        return

    prov_label = "Gemini" if _AI_PROVIDER_CFG == "gemini" else "AI"
    with st.spinner(f"Running {prov_label} prediction (data up to {origin_date} only)..."):
        ai_result = _dispatch_prediction(spi, ctx_bars)

    # Hard-fail guard: Gemini mode must never silently serve baseline output
    if _AI_PROVIDER_CFG == "gemini" and ai_result.get("source") != "gemini_stock_prediction_agent":
        st.error(
            f"RUNTIME WIRING ERROR: AI_PROVIDER=gemini but received "
            f"source={ai_result.get('source')}. "
            "The Gemini agent did not run. Check GEMINI_API_KEY and restart the server."
        )
        return

    st.session_state["ai_result"] = ai_result

    # Cache origin price per symbol for strike guidance
    _op = ai_result.get("origin_price_used")
    if _op and symbol:
        _pc = st.session_state.get("_symbol_price_cache", {})
        _pc[symbol.upper()] = round(float(_op), 2)
        st.session_state["_symbol_price_cache"] = _pc

    if ai_result.get("status") != "SUCCESS":
        st.session_state["error_msg"] = _friendly_ai_error(ai_result.get("error", ""))
        return

    ai_hash = ai_result.get("stock_prediction_input_hash", "")
    if ai_hash and ai_hash != input_hash:
        st.session_state["error_msg"] = (
            f"Hash mismatch: input_hash={input_hash}, AI hash={ai_hash}. Blocked."
        )
        return

    if run_type == "live_future_prediction":
        # Target date is in the future — actual validation is PENDING
        val_result = {
            "status":                      "PENDING",
            "source":                      "historical_stock_price_validation",
            "stock_prediction_input_hash": input_hash,
            "symbol":                      symbol,
            "requested_prediction_origin_date": origin_date,
            "effective_origin_date":       ai_result.get("effective_origin_date", origin_date),
            "requested_target_date":       target_date,
            "reason":                      "Target date is in the future. Actual validation will be available after the target date.",
            "validation_available_after":  target_date,
        }
        st.session_state["val_result"] = val_result
        saved, save_msg = save_pending_prediction(spi, ai_result)
        st.session_state["saved"]    = saved
        st.session_state["save_msg"] = save_msg
        return

    with st.spinner(f"Fetching actual price on {target_date}..."):
        val_result = run_stock_validation(spi, ai_result, hist)

    st.session_state["val_result"] = val_result

    val_hash = val_result.get("stock_prediction_input_hash", "")
    if val_hash and val_hash != input_hash:
        st.session_state["error_msg"] = (
            f"Validation hash mismatch: input_hash={input_hash}, val hash={val_hash}."
        )
        return

    if val_result.get("status") == "SUCCESS":
        saved, save_msg = save_stock_prediction_record(spi, ai_result, val_result)
    else:
        saved, save_msg = False, f"Validation failed -- not saved: {val_result.get('error', '')}"

    st.session_state["saved"]    = saved
    st.session_state["save_msg"] = save_msg


def _run_options_all(
    symbol, ctx_start, origin_date, horizon, capital, benchmark,
    val_mode, target_date, run_id, price_basis="close", options_params=None,
):
    """Options Strategy Validation orchestrator.

    Steps 1-6 are identical to _run_all (AI prediction with historical context).
    Step 7: options backtest runs from prediction_origin_date to target_date ONLY.
    The historical context window is NEVER used as the backtest range.
    """
    options_params = options_params or {}
    today = date.today()

    # ── Exact-strike server-side guard (belt-and-suspenders) ─────────────────
    _srv_is_exact = options_params.get("contract_selection_method") == "EXACT_STRIKE"
    _srv_strike   = options_params.get("requested_strike") or options_params.get("strike_price")
    if _srv_is_exact and not _srv_strike:
        st.session_state["opts_result"] = {
            "status":                "REVIEW_REQUIRED",
            "reason":                "EXACT_STRIKE mode requires a valid strike price. Run blocked on server side.",
            "backtest_status":       "NOT_RUN",
            "exact_validation_status": "INVALID_INPUT",
            "accuracy_saved":        False,
            "accuracy_skip_reason":  "MISSING_STRIKE_SERVER_GUARD",
        }
        st.session_state["saved"]    = False
        st.session_state["save_msg"] = "NOT SAVED — exact strike missing (server guard)"
        return

    # Detect run type
    try:
        _target_dt = datetime.strptime(target_date, "%Y-%m-%d").date()
    except Exception:
        _target_dt = today
    run_type = "live_future_prediction" if _target_dt > today else "historical_validation"

    spi = {
        "symbol":                        symbol,
        "historical_context_start_date": ctx_start,
        "prediction_origin_date":        origin_date,
        "decision_horizon_days":         horizon,
        "target_date":                   target_date,
        "initial_capital":               capital,
        "benchmark":                     benchmark,
        "validation_mode":               val_mode,
        "price_basis":                   price_basis,
        "options_params":                options_params,  # pass strategy to AI
    }

    valid, err = validate_stock_prediction_input(spi)
    if not valid:
        st.session_state["error_msg"] = f"Input validation failed: {err}"
        return

    input_hash = build_stock_prediction_hash(spi)
    _run_ts2 = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    _tt_cred_source = (
        "REFRESH_TOKEN" if os.getenv("TASTYTRADE_REFRESH_TOKEN")
        else "ACCESS_TOKEN" if os.getenv("TASTYTRADE_ACCESS_TOKEN")
        else "MISSING"
    )
    _run_ctx2 = {
        "run_id":     run_id,
        "input_hash": input_hash,
        "mode":       "options",
        "symbol":     symbol,
        "timestamp_utc": _run_ts2,
        "provider_status": {
            "historical_data":   "PENDING",
            "rapidapi":          "PENDING",
            "tastytrade":        "PENDING",
            "tastytrade_cred":   _tt_cred_source,
            "gemini":            "PENDING",
        },
        "tastytrade_debug": {
            "credential_source":     _tt_cred_source,
            "token_refresh_status":  "NOT_RUN",
            "customer_check_status": "NOT_RUN",
            "backtest_create":       "NOT_RUN",
            "backtest_poll":         "NOT_RUN",
        },
        "validation_status": "STARTED",
        "errors": [],
        "accuracy_eligible": False,
    }
    st.session_state.update({
        "spi":          spi,
        "input_hash":   input_hash,
        "run_type":     run_type,
        "run_ts":       _run_ts2,
        "opts_params":  options_params,
        "run_context":  _run_ctx2,
    })

    _today_str2 = today.strftime("%Y-%m-%d")

    with st.spinner(f"Fetching historical prices for {symbol} (trying multiple providers)..."):
        hist, hist_err, _hist_cov2 = fetch_price_history_for_range(
            symbol, ctx_start, _today_str2
        )

    _run_ctx2["provider_chain"]      = _hist_cov2.get("providers_tried", [])
    _run_ctx2["hist_provider"]       = _hist_cov2.get("provider", "unknown")
    _run_ctx2["effective_ctx_start"] = _hist_cov2.get("effective_start") or ctx_start
    _run_ctx2["requested_ctx_start"] = ctx_start
    _run_ctx2["ctx_coverage_status"] = _hist_cov2.get("coverage_status", "FAILED")
    _eff2 = _run_ctx2["effective_ctx_start"]
    if _eff2 > ctx_start:
        try:
            _gap2 = (datetime.strptime(_eff2, "%Y-%m-%d").date()
                     - datetime.strptime(ctx_start, "%Y-%m-%d").date()).days
        except Exception:
            _gap2 = 0
        _run_ctx2["ctx_start_gap_days"] = _gap2
    else:
        _run_ctx2["ctx_start_gap_days"] = 0

    if not hist:
        _run_ctx2["provider_status"]["historical_data"] = "FAILED"
        _run_ctx2["provider_status"]["gemini"]          = "NOT_RUN"
        _run_ctx2["provider_status"]["tastytrade"]      = "NOT_RUN"
        _run_ctx2["validation_status"] = "HISTORICAL_DATA_UNAVAILABLE"
        _run_ctx2["errors"].append(hist_err or "Historical price data unavailable")
        st.session_state["run_context"] = _run_ctx2
        st.session_state["error_msg"]   = hist_err
        _log_invalid_run(run_id, input_hash, symbol, "options",
                         "HISTORICAL_DATA_UNAVAILABLE", hist_err or "")
        return

    st.session_state["price_hist"]  = hist
    st.session_state["ctx_summary"] = get_context_summary(hist, ctx_start, origin_date)

    # ── TRUTH GATE FIRST: auth check before any Tastytrade health call ────────
    # This ensures that when FORCE_DISABLE_TASTYTRADE=true, we never call the
    # real health check and store stale "success" data in tastytrade_health.
    with st.spinner("Verifying Tastytrade auth (live API check)..."):
        from verify_tastytrade_auth_truth import verify_tastytrade_auth_truth as _tt_auth_check
        _tt_truth_early = _tt_auth_check()
    _run_ctx2["tastytrade_auth_truth"] = _tt_truth_early
    # Merge into tastytrade_debug immediately
    _run_ctx2["tastytrade_debug"]["credential_source"]     = _tt_truth_early["credential_source"]
    _run_ctx2["tastytrade_debug"]["token_refresh_status"]  = _tt_truth_early["token_refresh_status"]
    _run_ctx2["tastytrade_debug"]["customer_check_status"] = _tt_truth_early["customer_check_status"]
    _run_ctx2["tastytrade_debug"]["auth_http_status"]      = _tt_truth_early["auth_http_status"]
    _run_ctx2["tastytrade_debug"]["backtest_allowed"]      = _tt_truth_early["backtest_allowed"]
    _run_ctx2["tastytrade_debug"]["auth_truth_reason"]     = _tt_truth_early["reason"]
    _run_ctx2["provider_status"]["tastytrade"] = (
        "FORCE_DISABLED"  if _tt_truth_early["credential_source"] == "MISSING_FORCED" else
        "NOT_CONFIGURED"  if _tt_truth_early["credential_source"] == "MISSING" else
        "AUTH_FAILED"     if not _tt_truth_early["backtest_allowed"] else
        "AUTH_OK"
    )
    st.session_state["run_context"] = _run_ctx2

    # ── Paid API health checks ────────────────────────────────────────────────
    # RapidAPI: always call (independent of TT auth)
    # Tastytrade: ONLY call _tastytrade_hc() when auth truth already confirmed OK
    # This prevents stale "customer verified YES" appearing when force-disabled.
    with st.spinner("Calling RapidAPI health check..."):
        _rapi_result2 = _rapidapi_hc()
    st.session_state["rapidapi_health"] = _rapi_result2

    if _tt_truth_early["backtest_allowed"]:
        # Auth confirmed live — run health check for additional telemetry
        _tt_result2 = _tastytrade_hc()
    else:
        # Auth blocked — do NOT call real health check; synthesize a "not called" result
        _cred_src_e  = _tt_truth_early["credential_source"]
        _tt_result2 = {
            "called":               False,
            "endpoint":             "NOT_CALLED",
            "http_status":          None,
            "latency_ms":           None,
            "token_refreshed":      False,
            "customer_verified":    False,
            "refresh_present":      _tt_truth_early["refresh_token_present"],
            "role":                 "authentication_and_account_health_check",
            "used_in_stock_mode":   False,
            "used_in_options_mode": True,
            "error":                _tt_truth_early["reason"],
            "blocked_by":           _cred_src_e,
            "backtest_allowed":     False,
        }
    st.session_state["tastytrade_health"] = _tt_result2

    # Strict two-bound context filter
    ctx_bars = [b for b in hist if ctx_start <= b["date"] <= origin_date]
    if not ctx_bars:
        st.session_state["error_msg"] = (
            f"No price bars found between {ctx_start} and {origin_date}."
        )
        return

    # Pre-flight context validation
    if ctx_bars[0]["date"] < ctx_start:
        st.session_state["error_msg"] = (
            f"AI context leakage: first bar {ctx_bars[0]['date']} before {ctx_start}. Blocked."
        )
        return

    # ── CONTEXT-PERIOD TT OPTIMIZER — runs BEFORE Gemini so AI learns from real options data ──
    # Backtests the HISTORICAL CONTEXT window (ctx_start → origin_date) with different
    # parameter combos. Results are injected into the Gemini prompt so the AI recommends
    # parameters that ACTUALLY worked historically — not just technical indicator guesses.
    _ctx_tt_ok = (
        _TT_AVAILABLE and
        os.getenv("FORCE_DISABLE_TASTYTRADE", "false").lower() != "true" and
        st.session_state.get("tastytrade_health", {}).get("backtest_allowed") is not False and
        _tt_truth_early.get("backtest_allowed", False)
    )
    if _ctx_tt_ok:
        try:
            _cdir_raw  = str(options_params.get("direction", "Buy")).lower()
            _cdir_api  = "long"  if _cdir_raw in ("buy", "long")   else "short"
            _ctype_raw = str(options_params.get("opt_type", "Call")).lower()
            _ctype_api = "call"  if _ctype_raw == "call"           else "put"
            _cqty      = min(int(options_params.get("quantity") or 1), 5)
            _cfreq     = options_params.get("api_entry_frequency", "every day")
            _cexit     = options_params.get("exit_rule", "Exit at target date")
            _ctp_u     = options_params.get("take_profit_pct")
            _csl_u     = options_params.get("stop_loss_pct")

            # 6-combo grid — covers the key delta/DTE spread without hammering TT rate limits.
            # Kept to 6 calls so the main options backtest never gets a 429 rate-limit error.
            if _cdir_api == "short":
                _ctx_combos = [
                    (15, 21, 50, 150),   # low delta / mid DTE
                    (25, 14, 25, 100),   # low-mid delta / short DTE
                    (30, 21, 50, 150),   # mid delta / mid DTE
                    (35, 21, 75, 200),   # mid-high delta / mid DTE
                    (45, 14, 50, 150),   # high delta / short DTE
                    (45, 21, 75, 200),   # high delta / mid DTE
                ]
            else:
                _ctx_combos = [
                    (25, 14, 25, 10),    # low delta / short DTE
                    (35, 14, 25, 10),    # mid delta / short DTE
                    (40, 21, 30, 15),    # mid-high delta / mid DTE
                    (45, 14, 20, 10),    # near-ATM / short DTE
                    (50, 21, 35, 15),    # near-ATM / mid DTE
                    (55, 21, 30, 15),    # ATM / mid DTE
                ]

            _ctx_rows = []
            with st.spinner(
                f"🧠 AI Learning Phase — testing {len(_ctx_combos)} real TT backtests on "
                f"context period ({ctx_start} → {origin_date}) to teach AI what actually worked…"
            ):
                for _cci, (_ccd, _ccdte, _cctp, _ccsl) in enumerate(_ctx_combos):
                    if _cci > 0:
                        time.sleep(3)   # 3-second gap between calls to stay within TT rate limit
                    try:
                        _cc_leg = {
                            "type": "equity-option",
                            "direction": _cdir_api,
                            "quantity": _cqty,
                            "side": _ctype_api,
                            "daysUntilExpiration": _ccdte,
                            "strikeSelection": "delta",
                            "delta": _ccd,
                        }
                        _cc_payload = _tt_build_legs(
                            symbol=symbol,
                            start_date=ctx_start,
                            end_date=origin_date,
                            legs=[_cc_leg],
                            entry_frequency=_cfreq,
                            exit_rule=_cexit,
                            stop_loss_pct=float(_ccsl),
                            take_profit_pct=float(_cctp),
                        )
                        _cc_id, _ = _tt_create_backtest(_cc_payload)
                        if _cc_id:
                            _cc_data, _ = _tt_poll_backtest(_cc_id)
                            if _cc_data:
                                _cc_r   = _cc_data.get("results") or {}
                                _cc_st  = _cc_r.get("statistics") or {}
                                _cc_tr  = _cc_r.get("trials") or []
                                _cc_n   = int(_cc_st.get("Number of trades") or len(_cc_tr) or 0)
                                _cc_pnl = float(_cc_st.get("Total profit/loss") or 0)
                                _cc_wrr = float(_cc_st.get("Win percentage") or 0)
                                _cc_wr  = _cc_wrr / 100.0 if _cc_wrr > 1.0 else _cc_wrr
                                _ctx_rows.append({
                                    "delta": _ccd, "dte": _ccdte,
                                    "tp_pct": _cctp, "sl_pct": _ccsl,
                                    "total_pnl": _cc_pnl,
                                    "win_rate": _cc_wr,
                                    "trades": _cc_n,
                                    "avg_pnl": _cc_pnl / _cc_n if _cc_n else 0,
                                    "no_data": _cc_n == 0,
                                })
                            else:
                                _ctx_rows.append({
                                    "delta": _ccd, "dte": _ccdte,
                                    "tp_pct": _cctp, "sl_pct": _ccsl,
                                    "total_pnl": 0, "win_rate": 0,
                                    "trades": 0, "avg_pnl": 0, "no_data": True,
                                })
                        else:
                            _ctx_rows.append({
                                "delta": _ccd, "dte": _ccdte,
                                "tp_pct": _cctp, "sl_pct": _ccsl,
                                "total_pnl": 0, "win_rate": 0,
                                "trades": 0, "avg_pnl": 0, "no_data": True,
                            })
                    except Exception:
                        _ctx_rows.append({
                            "delta": _ccd, "dte": _ccdte,
                            "tp_pct": _cctp, "sl_pct": _ccsl,
                            "total_pnl": 0, "win_rate": 0,
                            "trades": 0, "avg_pnl": 0, "no_data": True,
                        })

            # Fetch current IV from yfinance (best-effort)
            _ctx_iv = None
            try:
                import yfinance as _yf_iv_mod
                _yft = _yf_iv_mod.Ticker(symbol)
                _yft_exps = _yft.options
                if _yft_exps:
                    _yft_chain = _yft.option_chain(_yft_exps[0])
                    _iv_c = float(_yft_chain.calls["impliedVolatility"].dropna().mean()) if len(_yft_chain.calls) else 0
                    _iv_p = float(_yft_chain.puts["impliedVolatility"].dropna().mean())  if len(_yft_chain.puts)  else 0
                    _ctx_iv = round((_iv_c + _iv_p) / 2, 4) if (_iv_c + _iv_p) > 0 else None
            except Exception:
                _ctx_iv = None

            # Store in spi so Gemini receives the real options intelligence
            if _ctx_rows:
                _ctx_valid_rows = sorted(
                    [r for r in _ctx_rows if not r.get("no_data")],
                    key=lambda x: x["total_pnl"], reverse=True
                )
                _ctx_empty_rows = [r for r in _ctx_rows if r.get("no_data")]
                spi["tt_context_optimizer"] = {
                    "results":      _ctx_valid_rows + _ctx_empty_rows,
                    "ctx_start":    ctx_start,
                    "origin_date":  origin_date,
                    "direction":    _cdir_api,
                    "opt_type":     _ctype_api,
                    "current_iv":   _ctx_iv,
                    "total_combos": len(_ctx_combos),
                    "valid_combos": len(_ctx_valid_rows),
                }
                st.session_state["ctx_optimizer_results"] = spi["tt_context_optimizer"]
        except Exception:
            pass  # Context optimizer is optional — never block the main run
        else:
            # Cooldown after optimizer — let TT API rate-limit window reset before main backtest
            time.sleep(5)

    prov_label2 = "Gemini" if _AI_PROVIDER_CFG == "gemini" else "AI"
    with st.spinner(f"Running {prov_label2} prediction (predicting forward from {origin_date}) — now informed by real options data…"):
        ai_result = _dispatch_prediction(spi, ctx_bars)

    # Hard-fail guard: Gemini mode must never silently serve baseline output
    if _AI_PROVIDER_CFG == "gemini" and ai_result.get("source") != "gemini_stock_prediction_agent":
        st.error(
            f"RUNTIME WIRING ERROR: AI_PROVIDER=gemini but received "
            f"source={ai_result.get('source')}. "
            "The Gemini agent did not run. Check GEMINI_API_KEY and restart the server."
        )
        return

    st.session_state["ai_result"] = ai_result

    if ai_result.get("status") != "SUCCESS":
        st.session_state["error_msg"] = _friendly_ai_error(ai_result.get("error", ""))
        return

    ai_hash = ai_result.get("stock_prediction_input_hash", "")
    if ai_hash and ai_hash != input_hash:
        st.session_state["error_msg"] = (
            f"Hash mismatch: input_hash={input_hash}, AI hash={ai_hash}."
        )
        return

    # Step 7 -- Options backtest: ctx_start → target_date (full study window)
    # Uses historical_context_start_date as backtest start so TastyTrade has enough
    # history to find matching contracts (mirrors TastyTrade website behaviour).
    if run_type == "live_future_prediction":
        # Target is future — backtest and actual validation are PENDING
        opts_result: Dict[str, Any] = {
            "status":  "PENDING",
            "reason":  "Target date is in the future. Backtest will be available after target date.",
            "backtest_range": f"{origin_date} to {target_date}",
        }
        val_result = {
            "status":                      "PENDING",
            "source":                      "historical_stock_price_validation",
            "stock_prediction_input_hash": input_hash,
            "symbol":                      symbol,
            "requested_prediction_origin_date": origin_date,
            "effective_origin_date":       ai_result.get("effective_origin_date", origin_date),
            "requested_target_date":       target_date,
            "reason":                      "Target date is in the future.",
            "validation_available_after":  target_date,
        }
        st.session_state["opts_result"] = opts_result
        st.session_state["val_result"]  = val_result
        saved, save_msg = save_pending_prediction(spi, ai_result)
        st.session_state["saved"]    = saved
        st.session_state["save_msg"] = save_msg
        return

    _is_exact_mode   = options_params.get("contract_selection_method") == "EXACT_STRIKE"
    _confirmed_strike = options_params.get("requested_strike") or options_params.get("strike_price")
    _confirmed_expiry = options_params.get("expiry_date")

    # ── Strike distance sanity check ─────────────────────────────────────────
    _strike_dist_pct    = None
    _strike_dist_status = "NOT_APPLICABLE"
    _strike_dist_warn   = ""
    if _is_exact_mode and _confirmed_strike:
        _origin_bars_chk = [b for b in hist if b.get("date", "") <= origin_date]
        _origin_px_chk   = float(
            (_origin_bars_chk[-1].get("open") or _origin_bars_chk[-1].get("close"))
            if _origin_bars_chk else 0
        )
        _strike_f = float(_confirmed_strike)
        if _origin_px_chk > 0:
            # Cache underlying price so strike guidance can show it before next run
            _price_cache = st.session_state.get("_symbol_price_cache", {})
            _price_cache[symbol.upper()] = round(_origin_px_chk, 2)
            st.session_state["_symbol_price_cache"] = _price_cache
            _strike_dist_pct = round(((_strike_f - _origin_px_chk) / _origin_px_chk) * 100, 2)
            if abs(_strike_dist_pct) > 200:
                _strike_dist_status = "EXTREME_STRIKE_DISTANCE"
                _strike_dist_warn = (
                    f"EXTREME_STRIKE_DISTANCE — requested strike {_strike_f:g} is "
                    f"{abs(_strike_dist_pct):.0f}% from underlying price {_origin_px_chk:.2f}. "
                    "Exact contract likely unavailable."
                )
            elif abs(_strike_dist_pct) > 50:
                _strike_dist_status = "FAR_STRIKE"
                _strike_dist_warn = (
                    f"FAR_STRIKE — requested strike {_strike_f:g} is "
                    f"{abs(_strike_dist_pct):.0f}% from underlying price {_origin_px_chk:.2f}."
                )
            elif abs(_strike_dist_pct) > 20:
                _strike_dist_status = "FAR_OTM_OR_ITM_STRIKE"
            else:
                _strike_dist_status = "NORMAL"

    # ── DTE/Expiry consistency check ─────────────────────────────────────────
    _computed_dte_from_expiry = None
    _effective_dte_used       = int(options_params.get("dte", 45))
    _dte_expiry_status        = "NOT_APPLICABLE"
    _dte_expiry_warn          = ""
    if _confirmed_expiry:
        try:
            _origin_dt_obj        = datetime.strptime(origin_date, "%Y-%m-%d").date()
            _expiry_dt_obj        = datetime.strptime(_confirmed_expiry, "%Y-%m-%d").date()
            _computed_dte_from_expiry = (_expiry_dt_obj - _origin_dt_obj).days
            _effective_dte_used   = _computed_dte_from_expiry
            _user_dte_v           = int(options_params.get("dte", 45))
            if abs(_computed_dte_from_expiry - _user_dte_v) > 3:
                _dte_expiry_status = "DTE_EXPIRY_MISMATCH"
                _dte_expiry_warn   = (
                    f"User DTE {_user_dte_v} does not match expiry-derived DTE "
                    f"{_computed_dte_from_expiry}. "
                    "Exact contract mode uses expiry date as source of truth."
                )
            else:
                _dte_expiry_status = "CONSISTENT"
        except Exception:
            _dte_expiry_status = "CANNOT_COMPUTE"

    opts_result: Dict[str, Any] = {
        "status":                  "SKIPPED",
        "backtest_range":          f"{origin_date} to {target_date}",
        "note":                    "Options backtest window = prediction window (origin_date to target_date)",
        "strike_selection":        options_params.get("strike_selection", "delta"),
        "contract_method":         options_params.get("contract_selection_method", "DELTA_SELECTION"),
        "requested_strike":        _confirmed_strike,
        "expiry_date":             _confirmed_expiry,
        "delta_ui":                options_params.get("delta_ui"),
        "dte":                     options_params.get("dte", 45),
        "effective_dte_used":      _effective_dte_used,
        "computed_dte_from_expiry": _computed_dte_from_expiry,
        "dte_adjustment_reason":   "EXPIRY_DATE_PRIORITY" if _confirmed_expiry else "USER_DTE",
        "dte_expiry_status":       _dte_expiry_status,
        "dte_expiry_mismatch_warning": _dte_expiry_warn,
        "strike_distance_pct":     _strike_dist_pct,
        "strike_distance_status":  _strike_dist_status,
        "strike_distance_warning": _strike_dist_warn,
        "quantity":                options_params.get("quantity", 1),
        "direction":               options_params.get("direction", ""),
        "opt_type":                options_params.get("opt_type", ""),
        "exact_validation_status": "NOT_RUN",
        "eligible_for_exact_accuracy": False if _is_exact_mode else None,
    }

    _accuracy_skip_reason = None

    if _TT_AVAILABLE:
        # ── Read auth truth already computed above (single source of truth) ───
        # verify_tastytrade_auth_truth() was called BEFORE the health checks.
        # Do NOT call it again — use the stored result to avoid second API round trip.
        _run_ctx2  = st.session_state.get("run_context", _run_ctx2)
        _tt_truth  = _run_ctx2.get("tastytrade_auth_truth", _tt_truth_early)
        _tt_dbg    = _run_ctx2.get("tastytrade_debug", {})

        if not _tt_truth["backtest_allowed"]:
            _cred_src = _tt_truth["credential_source"]
            opts_result["status"] = "BACKTEST_CREATE_FAILED"
            opts_result["error"]  = _tt_truth["reason"]
            _tt_dbg["backtest_create"] = (
                "BLOCKED_NO_CREDENTIALS"  if _cred_src in ("MISSING", "MISSING_FORCED") else
                "BLOCKED_AUTH_FAILED"
            )
            _run_ctx2["tastytrade_debug"] = _tt_dbg
            st.session_state["run_context"] = _run_ctx2
            _accuracy_skip_reason = (
                "TASTYTRADE_NOT_CONFIGURED" if _cred_src in ("MISSING", "MISSING_FORCED") else
                "TASTYTRADE_AUTH_FAILED"
            )
        else:
            # ── Extreme strike gate (block proxy by default if >200% distance) ──
            _allow_extreme_proxy = os.getenv("ALLOW_PROXY_FOR_EXTREME_STRIKE", "false").lower() == "true"
            if _strike_dist_status == "EXTREME_STRIKE_DISTANCE" and not _allow_extreme_proxy:
                opts_result.update({
                    "status":  "BLOCKED_EXTREME_STRIKE",
                    "error":   (
                        f"RUN BLOCKED — EXTREME_STRIKE_DISTANCE: requested strike "
                        f"{_confirmed_strike} is {abs(_strike_dist_pct or 0):.0f}% from underlying price. "
                        "Delta proxy disabled for extreme strikes by policy. "
                        "Set ALLOW_PROXY_FOR_EXTREME_STRIKE=true in .env to override."
                    ),
                    "exact_validation_status":     "UNSUPPORTED_BY_PROVIDER",
                    "eligible_for_exact_accuracy": False,
                    "accuracy_saved":              False,
                    "accuracy_skip_reason":        "EXTREME_STRIKE_DISTANCE_BLOCKED",
                })
                _accuracy_skip_reason = "EXTREME_STRIKE_DISTANCE_BLOCKED"
                st.session_state["opts_result"] = opts_result
            else:
              with st.spinner(
                f"Running options backtest ({origin_date} to {target_date})..."
              ):
                try:
                    direction_map = {"Buy": "long", "Sell": "short"}
                    type_map      = {"Call": "call", "Put": "put"}

                    # Build leg based on strike selection mode
                    _direction_api  = direction_map.get(options_params.get("direction", "Sell"), "short")
                    _type_api       = type_map.get(options_params.get("opt_type", "Put"), "put")
                    _qty            = options_params.get("quantity", 1)
                    _strike_sel_key = options_params.get("strike_selection", "delta").lower()

                    if _strike_sel_key == "delta":
                        _delta_ui = options_params.get("delta_ui") or options_params.get("delta")
                        if not _delta_ui:
                            opts_result.update({"status": "REVIEW_REQUIRED", "reason": "Delta value missing — cannot proceed."})
                            st.session_state["opts_result"] = opts_result
                            st.session_state["saved"]       = False
                            st.session_state["save_msg"]    = "NOT SAVED — delta value missing"
                            return opts_result
                        leg = {
                            "type":                "equity-option",
                            "direction":           _direction_api,
                            "quantity":            _qty,
                            "side":                _type_api,
                            "daysUntilExpiration": _effective_dte_used,
                            "strikeSelection":     "delta",
                            "delta":               int(_delta_ui),
                        }
                        opts_result["payload_dte"]      = _effective_dte_used
                        opts_result["validation_type"]  = "DELTA_SELECTION"
                        opts_result["fallback_used"]    = False
                        opts_result["eligible_for_exact_accuracy"] = False

                    elif _strike_sel_key == "percentage otm":
                        _otm = options_params.get("otm_pct") or 5.0
                        leg = {
                            "type":                "equity-option",
                            "direction":           _direction_api,
                            "quantity":            _qty,
                            "side":                _type_api,
                            "daysUntilExpiration": _effective_dte_used,
                            "strikeSelection":     "percentageOtm",
                            "percentageOtm":       float(_otm),
                        }
                        opts_result["payload_dte"]      = _effective_dte_used
                        opts_result["validation_type"]  = "PERCENTAGE_OTM_SELECTION"
                        opts_result["fallback_used"]    = False
                        opts_result["eligible_for_exact_accuracy"] = False

                    elif _strike_sel_key == "price offset from underlying":
                        _offset = options_params.get("price_offset") or 5.0
                        leg = {
                            "type":                "equity-option",
                            "direction":           _direction_api,
                            "quantity":            _qty,
                            "side":                _type_api,
                            "daysUntilExpiration": _effective_dte_used,
                            "strikeSelection":     "priceOffset",
                            "priceOffset":         float(_offset),
                        }
                        opts_result["payload_dte"]      = _effective_dte_used
                        opts_result["validation_type"]  = "PRICE_OFFSET_SELECTION"
                        opts_result["fallback_used"]    = False
                        opts_result["eligible_for_exact_accuracy"] = False

                    elif _strike_sel_key == "premium":
                        _prem = options_params.get("premium_target") or 1.0
                        leg = {
                            "type":                "equity-option",
                            "direction":           _direction_api,
                            "quantity":            _qty,
                            "side":                _type_api,
                            "daysUntilExpiration": _effective_dte_used,
                            "strikeSelection":     "premium",
                            "premium":             float(_prem),
                        }
                        opts_result["payload_dte"]      = _effective_dte_used
                        opts_result["validation_type"]  = "PREMIUM_SELECTION"
                        opts_result["fallback_used"]    = False
                        opts_result["eligible_for_exact_accuracy"] = False

                    else:
                        # Fallback to delta=30
                        leg = {
                            "type":                "equity-option",
                            "direction":           _direction_api,
                            "quantity":            _qty,
                            "side":                _type_api,
                            "daysUntilExpiration": _effective_dte_used,
                            "strikeSelection":     "delta",
                            "delta":               30,
                        }
                        opts_result["payload_dte"]      = _effective_dte_used
                        opts_result["validation_type"]  = "DELTA_SELECTION_FALLBACK"
                        opts_result["fallback_used"]    = False
                        opts_result["eligible_for_exact_accuracy"] = False

                    _api_entry_freq  = options_params.get("api_entry_frequency", "every day")
                    _exit_rule       = options_params.get("exit_rule", "Exit at target date")
                    _take_profit_pct = options_params.get("take_profit_pct")
                    _stop_loss_pct   = options_params.get("stop_loss_pct")
                    _exit_after_days = options_params.get("exit_after_days")
                    # Backtest runs over the prediction window (origin → target).
                    # AI learns from ctx_start → origin, then we validate over origin → target.
                    _bt_end_date = target_date
                    opts_result["backtest_range"] = f"{origin_date} → {_bt_end_date}"
                    payload = _tt_build_legs(
                        symbol=symbol,
                        start_date=origin_date,
                        end_date=_bt_end_date,
                        legs=[leg],
                        entry_frequency=_api_entry_freq,
                        exit_rule=_exit_rule,
                        stop_loss_pct=_stop_loss_pct,
                        take_profit_pct=_take_profit_pct,
                        exit_after_days=_exit_after_days,
                    )
                    st.session_state["backtest_payload"] = payload.to_dict()
                    backtest_id, err = _tt_create_backtest(payload)
                    # EXIT_COND_DROPPED means: backtest succeeded but exit conditions
                    # were auto-dropped (API rejected them); flag for UI warning.
                    _exit_cond_dropped = bool(err and err.startswith("EXIT_COND_DROPPED:"))
                    if _exit_cond_dropped:
                        opts_result["exit_cond_dropped"] = True
                        opts_result["exit_cond_drop_reason"] = err.split(":", 1)[-1][:300]
                    if backtest_id:
                        bt_data, poll_err = _tt_poll_backtest(backtest_id)
                        if bt_data:
                            _res_obj = bt_data.get("results") or {}
                            stats    = _res_obj.get("statistics") or {}
                            _trials  = _res_obj.get("trials") or []
                            _win_pct_raw = (
                                stats.get("Win percentage") or stats.get("winRate")
                                or stats.get("win_rate") or 0
                            )
                            _win_pct = float(_win_pct_raw or 0)
                            if _win_pct > 1.0:
                                _win_pct = _win_pct / 100.0
                            _total_pl = float(
                                stats.get("Total profit/loss") or stats.get("totalProfitLoss")
                                or stats.get("profit_loss") or 0
                            )
                            _n_trades = int(
                                stats.get("Number of trades") or stats.get("numTrades")
                                or stats.get("total_trades") or len(_trials) or 0
                            )
                            # Only store dollar P&L — never fall back to "Avg. return per trade"
                            # which is a percentage value and would display as "$-7.96" (wrong unit).
                            _avg_pnl_raw = float(
                                stats.get("Avg. profit/loss per trade")
                                or stats.get("avgProfitLoss") or stats.get("avg_pnl") or 0
                            )
                            _avg_pnl = _avg_pnl_raw if _avg_pnl_raw else (
                                _total_pl / _n_trades if _n_trades > 0 else 0
                            )

                            # Detect FLAT_NO_TRADES — 0 trades + $0 P&L = not a real validation
                            if _n_trades == 0 and _total_pl == 0.0:
                                opts_result.update({
                                    "status":       "FLAT_NO_TRADES",
                                    "backtest_id":  backtest_id,
                                    "win_rate":     0.0,
                                    "profit_loss":  0.0,
                                    "avg_pnl":      0.0,
                                    "total_trades": 0,
                                    "raw_stats":    stats,
                                    "note":         (
                                        "Tastytrade returned 0 trades and $0 P&L. "
                                        "No matching option contracts found in this window for the selected parameters. "
                                        "Not counted as validated."
                                    ),
                                })
                                if not _accuracy_skip_reason:
                                    _accuracy_skip_reason = "FLAT_NO_TRADES"
                            else:
                                _max_loss = float(
                                    stats.get("maxLoss") or stats.get("max_loss")
                                    or stats.get("Worst loss") or 0
                                )
                                _max_profit = float(
                                    stats.get("maxProfit") or stats.get("max_profit")
                                    or stats.get("Highest profit") or 0
                                )
                                opts_result.update({
                                    "status":       "SUCCESS",
                                    "backtest_id":  backtest_id,
                                    "win_rate":     _win_pct,
                                    "profit_loss":  _total_pl,
                                    "avg_pnl":      _avg_pnl,
                                    "total_trades": _n_trades,
                                    "max_loss":     _max_loss,
                                    "max_profit":   _max_profit,
                                    "raw_stats":    stats,
                                    "trials":       _trials,
                                    "raw_bt_data":  bt_data,
                                })
                                # ── BS Fallback: replace API result if it looks wrong ────────────
                                # For long options (Buy Call/Put), TastyTrade API often returns
                                # $0 prices and impossible P&L (loss > premium paid).
                                # Detect this and replace with Black-Scholes simulation.
                                try:
                                    from src.services.options_pricing_service import (
                                        simulate_options_strategy, needs_bs_fallback
                                    )
                                    _dir_for_bs = options_params.get("direction", "Buy")
                                    _sl_for_bs  = options_params.get("stop_loss_pct")
                                    if needs_bs_fallback(opts_result, _dir_for_bs, stop_loss_pct=_sl_for_bs):
                                        _bs_tp = options_params.get("take_profit_pct")
                                        _bs_sl = options_params.get("stop_loss_pct")
                                        _bs_delta = int(options_params.get("delta_ui") or options_params.get("delta") or 1)
                                        _bs_dte   = int(options_params.get("dte") or 15)
                                        _bs_qty   = int(options_params.get("quantity") or 1)
                                        _bs_type  = options_params.get("opt_type", "Call")
                                        _bs_freq  = options_params.get("api_entry_frequency", "every day")
                                        _bs_mult  = 100  # SPX/SPY/standard US options
                                        with st.spinner("TastyTrade API returned incorrect prices for this configuration — running Black-Scholes simulation..."):
                                            _bs_sim = simulate_options_strategy(
                                                hist_prices      = hist,
                                                start_date       = origin_date,
                                                end_date         = _bt_end_date,
                                                direction        = _dir_for_bs,
                                                option_type      = _bs_type,
                                                target_delta     = _bs_delta,
                                                dte              = _bs_dte,
                                                quantity         = _bs_qty,
                                                take_profit_pct  = _bs_tp,
                                                stop_loss_pct    = _bs_sl,
                                                entry_frequency  = _bs_freq,
                                                multiplier       = _bs_mult,
                                                symbol           = str(symbol),
                                                initial_capital  = float(capital),
                                            )
                                        if _bs_sim.get("status") == "SUCCESS":
                                            _bs_stats  = _bs_sim["statistics"]
                                            _bs_trials = _bs_sim["trials"]
                                            _bs_n      = len(_bs_trials)
                                            _bs_wins   = sum(1 for t in _bs_trials if float(t.get("profitLoss") or 0) > 0)
                                            _bs_wr     = _bs_wins / _bs_n if _bs_n > 0 else 0.0
                                            _bs_total  = float(_bs_stats.get("Total profit/loss") or 0)
                                            _bs_avg    = float(_bs_stats.get("Avg. profit/loss per trade") or 0)
                                            _bs_maxp   = float(_bs_stats.get("Highest profit") or 0)
                                            _bs_maxl   = float(_bs_stats.get("Worst loss") or 0)
                                            # Build a minimal raw_bt_data wrapper so downstream code can read daily settlements
                                            _bs_raw_bt = {
                                                "results": {
                                                    "trials":           _bs_trials,
                                                    "statistics":       _bs_stats,
                                                    "dailySettlements": _bs_sim.get("daily_settlements", []),
                                                    "transactions":     [],
                                                }
                                            }
                                            opts_result.update({
                                                "win_rate":     _bs_wr,
                                                "profit_loss":  _bs_total,
                                                "avg_pnl":      _bs_avg,
                                                "total_trades": _bs_n,
                                                "max_profit":   _bs_maxp,
                                                "max_loss":     _bs_maxl,
                                                "raw_stats":    _bs_stats,
                                                "trials":       _bs_trials,
                                                "raw_bt_data":  _bs_raw_bt,
                                                "bs_fallback":  True,
                                                "bs_sigma_pct": _bs_sim.get("sigma_used_pct"),
                                                "bs_source":    "black_scholes_simulation",
                                            })
                                except Exception as _bs_exc:
                                    opts_result["bs_fallback_error"] = str(_bs_exc)

                                # ── AUTO-CORRECT: TT API ignores tight SL for short options ──
                                # When TT API win rate > 75% on a Sell strategy with SL < 20%,
                                # the API silently ignored the stop loss. Automatically recompute
                                # using Black-Scholes with correct SL applied so the user sees
                                # accurate results inside the app — no redirect to website needed.
                                try:
                                    _sl_chk = options_params.get("stop_loss_pct")
                                    _dir_chk = options_params.get("direction", "")
                                    _wr_chk  = float(opts_result.get("win_rate") or 0)
                                    _wr_pct_chk = _wr_chk * 100.0 if _wr_chk <= 1.0 else _wr_chk
                                    if (
                                        not opts_result.get("bs_fallback")
                                        and _dir_chk.lower() in ("sell", "short")
                                        and _sl_chk is not None
                                        and float(_sl_chk) < 20.0
                                        and opts_result.get("status") == "SUCCESS"
                                        and _wr_pct_chk > 75.0
                                    ):
                                        # Auto-correct: run BS simulation with proper SL
                                        try:
                                            from src.services.options_pricing_service import simulate_options_strategy
                                            _sl_fix_tp   = options_params.get("take_profit_pct")
                                            _sl_fix_sl   = float(_sl_chk)
                                            _sl_fix_d    = int(options_params.get("delta_ui") or options_params.get("delta") or 20)
                                            _sl_fix_dte  = int(options_params.get("dte") or 45)
                                            _sl_fix_qty  = int(options_params.get("quantity") or 1)
                                            _sl_fix_type = options_params.get("opt_type", "Put")
                                            _sl_fix_freq = options_params.get("api_entry_frequency", "every day")
                                            with st.spinner("TastyTrade API ignored your Stop Loss — auto-correcting results with Black-Scholes simulation..."):
                                                _sl_fix_sim = simulate_options_strategy(
                                                    hist_prices     = hist,
                                                    start_date      = origin_date,
                                                    end_date        = _bt_end_date,
                                                    direction       = _dir_chk,
                                                    option_type     = _sl_fix_type,
                                                    target_delta    = _sl_fix_d,
                                                    dte             = _sl_fix_dte,
                                                    quantity        = _sl_fix_qty,
                                                    take_profit_pct = _sl_fix_tp,
                                                    stop_loss_pct   = _sl_fix_sl,
                                                    entry_frequency = _sl_fix_freq,
                                                    multiplier      = 100,
                                                    symbol          = str(symbol),
                                                    initial_capital = float(capital),
                                                )
                                            if _sl_fix_sim.get("status") == "SUCCESS":
                                                _slfix_stats  = _sl_fix_sim["statistics"]
                                                _slfix_trials = _sl_fix_sim["trials"]
                                                _slfix_n      = len(_slfix_trials)
                                                _slfix_wins   = sum(1 for t in _slfix_trials if float(t.get("profitLoss") or 0) > 0)
                                                _slfix_wr     = _slfix_wins / _slfix_n if _slfix_n > 0 else 0.0
                                                _slfix_total  = float(_slfix_stats.get("Total profit/loss") or 0)
                                                _slfix_avg    = float(_slfix_stats.get("Avg. profit/loss per trade") or 0)
                                                _slfix_maxp   = float(_slfix_stats.get("Highest profit") or 0)
                                                _slfix_maxl   = float(_slfix_stats.get("Worst loss") or 0)
                                                _slfix_raw_bt = {
                                                    "results": {
                                                        "trials":           _slfix_trials,
                                                        "statistics":       _slfix_stats,
                                                        "dailySettlements": _sl_fix_sim.get("daily_settlements", []),
                                                        "transactions":     [],
                                                    }
                                                }
                                                opts_result.update({
                                                    "win_rate":           _slfix_wr,
                                                    "profit_loss":        _slfix_total,
                                                    "avg_pnl":            _slfix_avg,
                                                    "total_trades":       _slfix_n,
                                                    "max_profit":         _slfix_maxp,
                                                    "max_loss":           _slfix_maxl,
                                                    "raw_stats":          _slfix_stats,
                                                    "trials":             _slfix_trials,
                                                    "raw_bt_data":        _slfix_raw_bt,
                                                    "bs_fallback":        True,
                                                    "bs_sl_corrected":    True,
                                                    "bs_sl_value":        _sl_fix_sl,
                                                    "bs_sigma_pct":       _sl_fix_sim.get("sigma_used_pct"),
                                                    "bs_source":          "black_scholes_sl_correction",
                                                    "tt_api_sl_not_applied": False,
                                                })
                                        except Exception as _slfix_exc:
                                            opts_result["tt_api_sl_not_applied"] = True
                                            opts_result["tt_api_sl_value"] = float(_sl_chk)
                                except Exception:
                                    pass

                                # ── AI STRATEGY SECOND BACKTEST ──────────────────────────────
                                # If AI recommended a different strategy, run a second backtest
                                # so the user can compare their strategy vs AI strategy performance.
                                try:
                                    _ai_rtc = ai_result.get("recommended_trade_config") or {}
                                    _ai_action_raw  = str(_ai_rtc.get("action") or "").strip()
                                    _ai_delta_raw   = _ai_rtc.get("suggested_delta")
                                    _ai_dte_raw     = _ai_rtc.get("suggested_dte")
                                    _ai_qty_raw     = _ai_rtc.get("suggested_quantity")
                                    _ai_tp_raw      = _ai_rtc.get("suggested_take_profit_pct")
                                    _ai_sl_raw      = _ai_rtc.get("suggested_stop_loss_pct")

                                    # Parse AI action into direction/type
                                    _ai_action_map = {
                                        "buy call":  ("long",  "call"),
                                        "buy put":   ("long",  "put"),
                                        "sell call": ("short", "call"),
                                        "sell put":  ("short", "put"),
                                    }
                                    _ai_dir_type = _ai_action_map.get(_ai_action_raw.lower())

                                    # Always use USER's delta/DTE/qty/TP/SL — only take direction from AI.
                                    # Gemini's suggested_delta/dte are often garbage defaults (30/30).
                                    _user_delta = int(options_params.get("delta_ui") or options_params.get("delta") or 0)
                                    _user_dte   = int(_effective_dte_used or options_params.get("dte") or 0)
                                    _user_qty   = int(options_params.get("quantity") or 1)
                                    _user_dir   = direction_map.get(options_params.get("direction", "Sell"), "short")
                                    _user_type  = type_map.get(options_params.get("opt_type", "Put"), "put")

                                    _ai_delta_int = _user_delta   # keep user's delta always
                                    _ai_dte_int   = _user_dte     # keep user's DTE always
                                    _ai_qty_int   = _user_qty     # keep user's quantity always
                                    _ai_tp_use    = _take_profit_pct  # keep user's TP always
                                    _ai_sl_use    = _stop_loss_pct    # keep user's SL always

                                    # Determine the AI-directed trade to validate.
                                    # Primary source: recommended_trade_config.action (what the AI actually recommends).
                                    # Fallback: derive from BUY/SELL decision only when RTC action is empty.
                                    # Never override the RTC recommendation — that IS the AI's trade signal.
                                    _ai_decision_upper = str(ai_result.get("decision", "")).upper()
                                    if _ai_dir_type:
                                        # RTC has a clear action (e.g. "Sell Put", "Buy Call") — use it exactly
                                        _signal_dir_type = _ai_dir_type
                                        # _ai_action_raw is already set correctly from _ai_rtc above
                                    elif _ai_decision_upper == "BUY":
                                        _signal_dir_type = ("long", "call")
                                        _ai_action_raw = "buy call"
                                    elif _ai_decision_upper == "SELL":
                                        _signal_dir_type = ("long", "put")
                                        _ai_action_raw = "buy put"
                                    else:
                                        _signal_dir_type = None  # HOLD with no RTC — skip AI backtest
                                    _dir_changed = _signal_dir_type and (
                                        _signal_dir_type[0] != _user_dir or _signal_dir_type[1] != _user_type
                                    )
                                    _ai_is_different = _signal_dir_type is not None and _dir_changed

                                    if _signal_dir_type and _ai_delta_int and _ai_dte_int:
                                        _ai_leg = {
                                            "type":                "equity-option",
                                            "direction":           _signal_dir_type[0],
                                            "quantity":            _ai_qty_int,
                                            "side":                _signal_dir_type[1],
                                            "daysUntilExpiration": _ai_dte_int,
                                            "strikeSelection":     "delta",
                                            "delta":               _ai_delta_int,
                                        }
                                        _ai_payload = _tt_build_legs(
                                            symbol=symbol,
                                            start_date=origin_date,
                                            end_date=_bt_end_date,
                                            legs=[_ai_leg],
                                            entry_frequency=_api_entry_freq,
                                            exit_rule=_exit_rule,
                                            stop_loss_pct=_ai_sl_use,
                                            take_profit_pct=_ai_tp_use,
                                        )
                                        with st.spinner(
                                            f"Running AI-recommended backtest "
                                            f"({_ai_action_raw}, Delta {_ai_delta_int}, DTE {_ai_dte_int})..."
                                        ):
                                            _ai_bt_id, _ai_bt_err = _tt_create_backtest(_ai_payload)
                                        if _ai_bt_id:
                                            _ai_bt_data, _ai_poll_err = _tt_poll_backtest(_ai_bt_id)
                                            if _ai_bt_data:
                                                _ai_res = _ai_bt_data.get("results") or {}
                                                _ai_stats = _ai_res.get("statistics") or {}
                                                _ai_trials = _ai_res.get("trials") or []
                                                _ai_wr_raw = float(
                                                    _ai_stats.get("Win percentage") or
                                                    _ai_stats.get("win_rate") or 0
                                                )
                                                _ai_wr = _ai_wr_raw / 100.0 if _ai_wr_raw > 1.0 else _ai_wr_raw
                                                _ai_pnl = float(
                                                    _ai_stats.get("Total profit/loss") or 0
                                                )
                                                _ai_n = int(
                                                    _ai_stats.get("Number of trades") or
                                                    len(_ai_trials) or 0
                                                )
                                                _ai_avg = _ai_pnl / _ai_n if _ai_n > 0 else 0
                                                _ai_bs_used = False

                                                # TT API returns $0 profitLoss for long (Buy) options.
                                                # Apply BS fallback to AI-directed trials just like we do
                                                # for the primary backtest — gives real per-trade P&L.
                                                if _signal_dir_type and _signal_dir_type[0] == "long":
                                                    try:
                                                        from src.services.options_pricing_service import (
                                                            simulate_options_strategy, needs_bs_fallback
                                                        )
                                                        _ai_chk = {
                                                            "win_rate": _ai_wr,
                                                            "profit_loss": _ai_pnl,
                                                            "trials": _ai_trials,
                                                            "total_trades": _ai_n,
                                                        }
                                                        _ai_opt_type_bs = "Call" if _signal_dir_type[1] == "call" else "Put"
                                                        if needs_bs_fallback(_ai_chk, "Buy", stop_loss_pct=_ai_sl_use):
                                                            _ai_bs_sim = simulate_options_strategy(
                                                                hist_prices     = hist,
                                                                start_date      = origin_date,
                                                                end_date        = _bt_end_date,
                                                                direction       = "Buy",
                                                                option_type     = _ai_opt_type_bs,
                                                                target_delta    = _ai_delta_int,
                                                                dte             = _ai_dte_int,
                                                                quantity        = _ai_qty_int,
                                                                take_profit_pct = _ai_tp_use,
                                                                stop_loss_pct   = _ai_sl_use,
                                                                entry_frequency = _api_entry_freq,
                                                                multiplier      = 100,
                                                                symbol          = str(symbol),
                                                                initial_capital = float(capital),
                                                            )
                                                            if _ai_bs_sim.get("status") == "SUCCESS":
                                                                _ai_trials = _ai_bs_sim["trials"]
                                                                _ai_stats_bs = _ai_bs_sim["statistics"]
                                                                _ai_pnl  = float(_ai_stats_bs.get("Total profit/loss") or 0)
                                                                _ai_n    = len(_ai_trials)
                                                                _ai_wins_bs = sum(1 for t in _ai_trials if float(t.get("profitLoss") or 0) > 0)
                                                                _ai_wr   = _ai_wins_bs / _ai_n if _ai_n else 0.0
                                                                _ai_avg  = _ai_pnl / _ai_n if _ai_n else 0
                                                                _ai_bs_used = True
                                                    except Exception:
                                                        pass

                                                opts_result["ai_strategy_result"] = {
                                                    "status":       "SUCCESS",
                                                    "action":       _ai_action_raw,
                                                    "delta":        _ai_delta_int,
                                                    "dte":          _ai_dte_int,
                                                    "quantity":     _ai_qty_int or 1,
                                                    "take_profit_pct": _ai_tp_use,
                                                    "stop_loss_pct":   _ai_sl_use,
                                                    "win_rate":     _ai_wr,
                                                    "profit_loss":  _ai_pnl,
                                                    "total_trades": _ai_n,
                                                    "avg_pnl":      _ai_avg,
                                                    "backtest_id":  _ai_bt_id,
                                                    "raw_stats":    _ai_stats,
                                                    "trials":       _ai_trials,
                                                    "bs_fallback":  _ai_bs_used,
                                                }
                                        else:
                                            opts_result["ai_strategy_result"] = {
                                                "status": "FAILED",
                                                "error":  _ai_bt_err or "Could not create AI strategy backtest",
                                            }
                                except Exception as _ai_bt_exc:
                                    opts_result["ai_strategy_result"] = {
                                        "status": "ERROR",
                                        "error":  str(_ai_bt_exc),
                                    }

                                # ── AI RECOMMENDED PARAMS BACKTEST ──────────────────────────
                                # Runs TT API with AI's actual recommended delta/DTE/qty/TP/SL
                                # so user sees: MY params result vs AI's recommended params result.
                                try:
                                    _ai_rtc2 = ai_result.get("recommended_trade_config") or {}
                                    _ai_rec_act = str(_ai_rtc2.get("action") or _ai_action_raw or "").strip().lower()
                                    _ai_rec_amap = {
                                        "buy call":  ("long",  "call"),
                                        "buy put":   ("long",  "put"),
                                        "sell call": ("short", "call"),
                                        "sell put":  ("short", "put"),
                                    }
                                    _ai_rec_dt = _ai_rec_amap.get(_ai_rec_act)
                                    if not _ai_rec_dt and _signal_dir_type:
                                        _ai_rec_dt = _signal_dir_type
                                        _ai_rec_act = _ai_action_raw
                                    if _ai_rec_dt:
                                        _rec_delta = int(_ai_rtc2.get("suggested_delta") or _user_delta or 30)
                                        _rec_dte   = int(_ai_rtc2.get("suggested_dte") or _user_dte or 21)
                                        _rec_qty   = min(int(_ai_rtc2.get("suggested_quantity") or _user_qty or 1), 10)
                                        _rec_tp_v  = _ai_rtc2.get("suggested_take_profit_pct")
                                        _rec_sl_v  = _ai_rtc2.get("suggested_stop_loss_pct")
                                        _rec_tp    = float(_rec_tp_v) if _rec_tp_v is not None else _take_profit_pct
                                        _rec_sl    = float(_rec_sl_v) if _rec_sl_v is not None else _stop_loss_pct
                                        # Only run if at least one param differs from user's
                                        _rec_differs = (
                                            _rec_delta != _user_delta or _rec_dte != _user_dte
                                            or _rec_qty != _user_qty
                                            or (_rec_tp or 0) != float(_take_profit_pct or 0)
                                            or (_rec_sl or 0) != float(_stop_loss_pct or 0)
                                            or _ai_rec_dt[0] != _user_dir or _ai_rec_dt[1] != _user_type
                                        )
                                        if _rec_differs:
                                            _rec_leg = {
                                                "type": "equity-option",
                                                "direction": _ai_rec_dt[0],
                                                "quantity": _rec_qty,
                                                "side": _ai_rec_dt[1],
                                                "daysUntilExpiration": _rec_dte,
                                                "strikeSelection": "delta",
                                                "delta": _rec_delta,
                                            }
                                            _rec_payload = _tt_build_legs(
                                                symbol=symbol, start_date=origin_date,
                                                end_date=_bt_end_date, legs=[_rec_leg],
                                                entry_frequency=_api_entry_freq,
                                                exit_rule=_exit_rule,
                                                stop_loss_pct=_rec_sl, take_profit_pct=_rec_tp,
                                            )
                                            with st.spinner(
                                                f"Running AI recommended backtest "
                                                f"({_ai_rec_act}, D{_rec_delta}, {_rec_dte}DTE, {_rec_qty}qty)..."
                                            ):
                                                _rec_bt_id, _rec_bt_err = _tt_create_backtest(_rec_payload)
                                            if _rec_bt_id:
                                                _rec_bt_data, _ = _tt_poll_backtest(_rec_bt_id)
                                                if _rec_bt_data:
                                                    _rec_res = _rec_bt_data.get("results") or {}
                                                    _rec_st  = _rec_res.get("statistics") or {}
                                                    _rec_tr  = _rec_res.get("trials") or []
                                                    _rec_wr_r = float(_rec_st.get("Win percentage") or _rec_st.get("winRate") or 0)
                                                    _rec_wr   = _rec_wr_r / 100.0 if _rec_wr_r > 1.0 else _rec_wr_r
                                                    _rec_pnl  = float(_rec_st.get("Total profit/loss") or 0)
                                                    _rec_n    = int(_rec_st.get("Number of trades") or len(_rec_tr) or 0)
                                                    opts_result["ai_rec_strategy_result"] = {
                                                        "status":          "SUCCESS",
                                                        "action":          _ai_rec_act,
                                                        "delta":           _rec_delta,
                                                        "dte":             _rec_dte,
                                                        "quantity":        _rec_qty,
                                                        "take_profit_pct": _rec_tp,
                                                        "stop_loss_pct":   _rec_sl,
                                                        "win_rate":        _rec_wr,
                                                        "profit_loss":     _rec_pnl,
                                                        "total_trades":    _rec_n,
                                                        "avg_pnl":         _rec_pnl / _rec_n if _rec_n else 0,
                                                        "backtest_id":     _rec_bt_id,
                                                    }
                                except Exception:
                                    pass  # AI rec backtest is optional — never break the main result

                                # ── STRATEGY OPTIMIZER COMPUTATION ──────────────────────────
                                # Auto-runs direction-aware TT API backtests across different
                                # delta/DTE/TP/SL combinations to find the historically best params.
                                try:
                                    # SELL strategies: delta 10-45 (selling OTM options)
                                    # BUY strategies: delta 20-55 (buying OTM to ATM options)
                                    if _user_dir == "short":
                                        _sopt_combos_run = [
                                            (10,  7, 25, 100), (10, 21, 50, 150),
                                            (15,  7, 25, 100), (15, 21, 50, 150),
                                            (20,  7, 25, 100), (20, 14, 50, 150), (20, 30, 75, 200),
                                            (25, 14, 25, 100), (25, 21, 50, 150), (25, 30, 75, 200),
                                            (30, 14, 50, 150), (30, 21, 50, 150), (30, 30, 75, 200),
                                            (35, 14, 25, 100), (35, 21, 50, 150),
                                            (40, 14, 50, 150), (40, 21, 75, 200),
                                            (45, 14, 25, 100), (45, 21, 50, 150),
                                            (50, 21, 50, 150),
                                        ]
                                    else:
                                        _sopt_combos_run = [
                                            (20,  7, 15,  5), (20, 14, 20,  8),
                                            (25,  7, 15,  5), (25, 14, 25, 10), (25, 21, 30, 15),
                                            (30, 14, 20,  8), (30, 21, 25, 10), (30, 30, 30, 15),
                                            (35, 14, 25, 10), (35, 21, 30, 15),
                                            (40, 14, 20,  8), (40, 21, 30, 15), (40, 30, 35, 20),
                                            (45, 14, 20, 10), (45, 21, 30, 15), (45, 30, 40, 20),
                                            (50, 14, 25, 10), (50, 21, 35, 15),
                                            (55, 21, 30, 15), (55, 30, 40, 20),
                                        ]
                                    _sopt_total = len(_sopt_combos_run)
                                    _sopt_run_rows = []
                                    with st.spinner(f"🔍 Strategy Optimizer — testing {_sopt_total} parameter combinations via TT API…"):
                                        for _sci, (_scd, _scdte, _sctp, _scsl) in enumerate(_sopt_combos_run):
                                            try:
                                                _sc_leg = {
                                                    "type": "equity-option",
                                                    "direction": _user_dir,
                                                    "quantity": min(_user_qty, 5),
                                                    "side": _user_type,
                                                    "daysUntilExpiration": _scdte,
                                                    "strikeSelection": "delta",
                                                    "delta": _scd,
                                                }
                                                _sc_payload = _tt_build_legs(
                                                    symbol=symbol,
                                                    start_date=origin_date,
                                                    end_date=_bt_end_date,
                                                    legs=[_sc_leg],
                                                    entry_frequency=_api_entry_freq,
                                                    exit_rule=_exit_rule,
                                                    stop_loss_pct=float(_scsl),
                                                    take_profit_pct=float(_sctp),
                                                )
                                                _sc_id, _ = _tt_create_backtest(_sc_payload)
                                                if _sc_id:
                                                    _sc_data, _ = _tt_poll_backtest(_sc_id)
                                                    if _sc_data:
                                                        _sc_r   = _sc_data.get("results") or {}
                                                        _sc_st  = _sc_r.get("statistics") or {}
                                                        _sc_tr  = _sc_r.get("trials") or []
                                                        _sc_n   = int(_sc_st.get("Number of trades") or len(_sc_tr) or 0)
                                                        _sc_pnl = float(_sc_st.get("Total profit/loss") or 0)
                                                        _sc_wrr = float(_sc_st.get("Win percentage") or 0)
                                                        _sc_wr  = _sc_wrr / 100.0 if _sc_wrr > 1.0 else _sc_wrr
                                                        _sopt_run_rows.append({
                                                            "delta": _scd, "dte": _scdte,
                                                            "tp_pct": _sctp, "sl_pct": _scsl,
                                                            "qty": min(_user_qty, 5),
                                                            "total_pnl": _sc_pnl,
                                                            "win_rate": _sc_wr,
                                                            "trades": _sc_n,
                                                            "avg_pnl": _sc_pnl / _sc_n if _sc_n else 0,
                                                            "no_data": _sc_n == 0,
                                                        })
                                            except Exception:
                                                _sopt_run_rows.append({
                                                    "delta": _scd, "dte": _scdte,
                                                    "tp_pct": _sctp, "sl_pct": _scsl,
                                                    "qty": min(_user_qty, 5),
                                                    "total_pnl": 0, "win_rate": 0,
                                                    "trades": 0, "avg_pnl": 0,
                                                    "no_data": True,
                                                })
                                    if _sopt_run_rows:
                                        # Sort: valid results by P&L first, then no-data rows at bottom
                                        _sopt_valid = [r for r in _sopt_run_rows if not r.get("no_data")]
                                        _sopt_empty = [r for r in _sopt_run_rows if r.get("no_data")]
                                        _sopt_valid.sort(key=lambda x: x["total_pnl"], reverse=True)
                                        opts_result["optimizer_results"] = _sopt_valid + _sopt_empty
                                        opts_result["optimizer_total_tested"] = _sopt_total
                                except Exception:
                                    pass  # Optimizer is optional — never break the main result

                        else:
                            opts_result["status"] = "BACKTEST_POLL_FAILED"
                            opts_result["error"]  = poll_err
                    else:
                        # Detect HTTP 429 rate limit from service error string
                        if err and err.startswith("RATE_LIMITED:429:"):
                            _retry_after_raw = err.split("retry_after=")[-1] if "retry_after=" in err else "unknown"
                            opts_result.update({
                                "status":              "RATE_LIMITED",
                                "error":               (
                                    f"Tastytrade rate limit reached (HTTP 429). "
                                    f"Retry-After: {_retry_after_raw}. "
                                    "Wait before retrying. Accuracy NOT saved."
                                ),
                                "http_status":         429,
                                "retry_after":         _retry_after_raw,
                                "accuracy_saved":      False,
                                "accuracy_skip_reason": "RATE_LIMITED_429",
                                "backtest_id":         None,
                            })
                            _accuracy_skip_reason = "RATE_LIMITED_429"
                        elif err and err.startswith("BACKTEST_HTTP_"):
                            # Parse structured error: BACKTEST_HTTP_400:body_text
                            _parts = err.split(":", 1)
                            _http_code_str = _parts[0].replace("BACKTEST_HTTP_", "")
                            _err_body_raw = _parts[1] if len(_parts) > 1 else ""
                            try:
                                _http_code_int = int(_http_code_str)
                            except ValueError:
                                _http_code_int = 0
                            opts_result.update({
                                "status":              "BACKTEST_CREATE_FAILED",
                                "http_status":         _http_code_int,
                                "error":               (
                                    f"Failed to create backtest: HTTP {_http_code_int}. "
                                    "Tastytrade authentication succeeded — this is a payload/provider validation issue, "
                                    "NOT a credential issue. Check payload, DTE, and strike parameters."
                                ),
                                "error_body":          _err_body_raw,
                                "accuracy_saved":      False,
                                "accuracy_skip_reason": f"BACKTEST_CREATE_FAILED_HTTP_{_http_code_int}",
                                "backtest_id":         None,
                            })
                            _accuracy_skip_reason = f"BACKTEST_CREATE_FAILED_HTTP_{_http_code_int}"
                        else:
                            opts_result["status"] = "BACKTEST_CREATE_FAILED"
                            opts_result["error"]  = err
                except Exception as exc:
                    opts_result["status"] = "ERROR"
                    opts_result["error"]  = f"{type(exc).__name__}: {exc}"
    else:
        opts_result["error"] = "Tastytrade backtester not available (import failed or credentials missing)"

    # ── BS Simulation when TastyTrade API failed entirely ──────────────────────
    # If the API couldn't run (auth fail, HTTP error, or credentials missing),
    # run a Black-Scholes simulation so the user still gets meaningful results.
    if opts_result.get("status") not in ("SUCCESS", "PENDING", "FLAT_NO_TRADES"):
        try:
            from src.services.options_pricing_service import simulate_options_strategy
            _dir_for_bs2 = options_params.get("direction", "Buy")
            _bs_tp2  = options_params.get("take_profit_pct")
            _bs_sl2  = options_params.get("stop_loss_pct")
            _bs_d2   = int(options_params.get("delta_ui") or options_params.get("delta") or 1)
            _bs_dte2 = int(options_params.get("dte") or 15)
            _bs_qty2 = int(options_params.get("quantity") or 1)
            _bs_t2   = options_params.get("opt_type", "Call")
            _bs_f2   = options_params.get("api_entry_frequency", "every day")
            with st.spinner("Running Black-Scholes options simulation (TastyTrade API unavailable)..."):
                _bs_sim2 = simulate_options_strategy(
                    hist_prices     = hist,
                    start_date      = origin_date,
                    end_date        = target_date,
                    direction       = _dir_for_bs2,
                    option_type     = _bs_t2,
                    target_delta    = _bs_d2,
                    dte             = _bs_dte2,
                    quantity        = _bs_qty2,
                    take_profit_pct = _bs_tp2,
                    stop_loss_pct   = _bs_sl2,
                    entry_frequency = _bs_f2,
                    multiplier      = 100,
                    symbol          = str(symbol),
                    initial_capital = float(capital),
                )
            if _bs_sim2.get("status") == "SUCCESS":
                _bs_s2  = _bs_sim2["statistics"]
                _bs_tr2 = _bs_sim2["trials"]
                _bs_n2  = len(_bs_tr2)
                _bs_w2  = sum(1 for t in _bs_tr2 if float(t.get("profitLoss") or 0) > 0)
                _prev_err      = opts_result.get("error", "")
                _prev_err_body = opts_result.get("error_body", "")
                opts_result.update({
                    "status":       "SUCCESS",
                    "win_rate":     _bs_w2 / _bs_n2 if _bs_n2 > 0 else 0.0,
                    "profit_loss":  float(_bs_s2.get("Total profit/loss") or 0),
                    "avg_pnl":      float(_bs_s2.get("Avg. profit/loss per trade") or 0),
                    "total_trades": _bs_n2,
                    "max_profit":   float(_bs_s2.get("Highest profit") or 0),
                    "max_loss":     float(_bs_s2.get("Worst loss") or 0),
                    "raw_stats":    _bs_s2,
                    "trials":       _bs_tr2,
                    "raw_bt_data":  {
                        "results": {
                            "trials":           _bs_tr2,
                            "statistics":       _bs_s2,
                            "dailySettlements": _bs_sim2.get("daily_settlements", []),
                            "transactions":     [],
                        }
                    },
                    "bs_fallback":        True,
                    "bs_fallback_reason": _prev_err,
                    "bs_fallback_body":   _prev_err_body,
                    "bs_sigma_pct":       _bs_sim2.get("sigma_used_pct"),
                    "bs_source":          "black_scholes_simulation",
                    "backtest_range":     f"{origin_date} → {target_date}",
                })
                _accuracy_skip_reason = "BS_SIMULATION_ONLY"
        except Exception as _bs_exc2:
            pass   # leave original error status intact

    # ── AI-Directed BS Comparison (always runs, TT-auth-independent) ──────────
    # Computes what the AI's recommended trade direction returns via Black-Scholes.
    # Runs regardless of TT auth status so the comparison table always shows.
    try:
        from datetime import datetime as _dt_cmp
        from src.services.options_pricing_service import simulate_options_strategy as _sop_c
        _ai_rtc_c  = (ai_result or {}).get("recommended_trade_config") or {}
        _ai_dec_c  = str((ai_result or {}).get("decision", "")).upper()
        _ai_rec_c  = str(_ai_rtc_c.get("action") or "").lower().strip()
        _ai_amap_c = {
            "buy call":  ("Buy", "Call"),
            "buy put":   ("Buy", "Put"),
            "sell call": ("Sell", "Call"),
            "sell put":  ("Sell", "Put"),
        }
        _ai_dir_c, _ai_type_c = _ai_amap_c.get(_ai_rec_c, (None, None))
        if not _ai_dir_c:
            if _ai_dec_c == "BUY":
                _ai_dir_c, _ai_type_c = "Buy", "Call"
            elif _ai_dec_c == "SELL":
                _ai_dir_c, _ai_type_c = "Buy", "Put"
        if _ai_dir_c and hist:
            # Cap DTE to available horizon so trades don't expire past target_date
            try:
                _avail_days = (_dt_cmp.strptime(target_date, "%Y-%m-%d") - _dt_cmp.strptime(origin_date, "%Y-%m-%d")).days
            except Exception:
                _avail_days = 30
            _ai_delta_c    = int(_ai_rtc_c.get("suggested_delta") or options_params.get("delta_ui") or options_params.get("delta") or 20)
            _ai_dte_raw    = int(_ai_rtc_c.get("suggested_dte") or options_params.get("dte") or 21)
            _ai_dte_c      = max(1, min(_ai_dte_raw, max(1, _avail_days - 1)))
            _ai_qty_c      = int(options_params.get("quantity") or 1)
            _ai_tp_c       = options_params.get("take_profit_pct")
            _ai_sl_c       = options_params.get("stop_loss_pct")
            _ai_freq_c     = options_params.get("api_entry_frequency", "every day")
            _ai_comp_sim = _sop_c(
                hist_prices=hist, start_date=origin_date, end_date=target_date,
                direction=_ai_dir_c, option_type=_ai_type_c,
                target_delta=_ai_delta_c, dte=_ai_dte_c, quantity=_ai_qty_c,
                take_profit_pct=_ai_tp_c, stop_loss_pct=_ai_sl_c,
                entry_frequency=_ai_freq_c, multiplier=100,
                symbol=str(symbol), initial_capital=float(capital),
            )
            # If capped DTE returned 0 trials, retry with user-configured DTE
            _ai_comp_tr = (_ai_comp_sim.get("trials") or []) if _ai_comp_sim.get("status") == "SUCCESS" else []
            if not _ai_comp_tr and _ai_dte_c != _ai_dte_raw:
                _ai_dte_fb = max(1, min(int(options_params.get("dte") or 21), max(1, _avail_days - 1)))
                _ai_comp_sim2 = _sop_c(
                    hist_prices=hist, start_date=origin_date, end_date=target_date,
                    direction=_ai_dir_c, option_type=_ai_type_c,
                    target_delta=_ai_delta_c, dte=_ai_dte_fb, quantity=_ai_qty_c,
                    take_profit_pct=_ai_tp_c, stop_loss_pct=_ai_sl_c,
                    entry_frequency=_ai_freq_c, multiplier=100,
                    symbol=str(symbol), initial_capital=float(capital),
                )
                if _ai_comp_sim2.get("status") == "SUCCESS" and _ai_comp_sim2.get("trials"):
                    _ai_comp_sim = _ai_comp_sim2
                    _ai_comp_tr  = _ai_comp_sim2["trials"]
                    _ai_dte_c    = _ai_dte_fb
            if _ai_comp_sim.get("status") == "SUCCESS" and _ai_comp_tr:
                _ai_comp_st  = _ai_comp_sim.get("statistics") or {}
                _ai_comp_n   = len(_ai_comp_tr)
                _ai_comp_w   = sum(1 for _t in _ai_comp_tr if float(_t.get("profitLoss") or 0) > 0)
                _ai_comp_pl  = float(
                    _ai_comp_st.get("Total profit/loss") or
                    sum(float(_t.get("profitLoss") or 0) for _t in _ai_comp_tr)
                )
                opts_result["ai_comparison_trials"]       = _ai_comp_tr
                opts_result["ai_comparison_action"]       = f"{_ai_dir_c} {_ai_type_c}"
                opts_result["ai_comparison_delta"]        = _ai_delta_c
                opts_result["ai_comparison_dte"]          = _ai_dte_c
                opts_result["ai_comparison_dte_original"] = _ai_dte_raw
                opts_result["ai_comparison_win_rate"]     = _ai_comp_w / _ai_comp_n if _ai_comp_n else 0.0
                opts_result["ai_comparison_profit_loss"]  = _ai_comp_pl
                opts_result["ai_comparison_total_trades"] = _ai_comp_n
                # Supplemental pass: fill dates beyond the AI-DTE window by running the same
                # AI direction with the user's actual DTE. This eliminates "No match" entries
                # when the AI-suggested DTE (e.g. 45) is larger than the user's DTE (e.g. 6).
                _usr_dte_sup = int(options_params.get("dte") or options_params.get("dte_ui") or 0)
                if _usr_dte_sup and _usr_dte_sup != _ai_dte_c:
                    try:
                        _ai_comp_sim_u = _sop_c(
                            hist_prices=hist, start_date=origin_date, end_date=target_date,
                            direction=_ai_dir_c, option_type=_ai_type_c,
                            target_delta=_ai_delta_c, dte=_usr_dte_sup, quantity=_ai_qty_c,
                            take_profit_pct=_ai_tp_c, stop_loss_pct=_ai_sl_c,
                            entry_frequency=_ai_freq_c, multiplier=100,
                            symbol=str(symbol), initial_capital=float(capital),
                        )
                        _ai_comp_tr_u = (_ai_comp_sim_u.get("trials") or []) if _ai_comp_sim_u.get("status") == "SUCCESS" else []
                        if _ai_comp_tr_u:
                            opts_result["ai_comparison_trials_user_dte"] = _ai_comp_tr_u
                            opts_result["ai_comparison_user_dte"] = _usr_dte_sup
                    except Exception:
                        pass
    except Exception as _cmp_exc:
        opts_result["ai_comparison_error"] = f"{type(_cmp_exc).__name__}: {_cmp_exc}"

    # ── Finalize accuracy_skip_reason in opts_result ──────────────────────────
    if _accuracy_skip_reason and not opts_result.get("accuracy_skip_reason"):
        opts_result["accuracy_skip_reason"] = _accuracy_skip_reason
    # Ensure accuracy_saved=false always has an explicit reason
    if not opts_result.get("accuracy_skip_reason") and opts_result.get("status") != "SUCCESS":
        opts_result["accuracy_skip_reason"] = f"BACKTEST_{opts_result.get('status', 'UNKNOWN')}"

    # ── Finalize backtest chain consistency ───────────────────────────────────
    _run_ctx2 = st.session_state.get("run_context", {})
    _tt_dbg2  = _run_ctx2.get("tastytrade_debug", {})
    _bs       = opts_result.get("status", "")
    if _bs == "FLAT_NO_TRADES" and opts_result.get("backtest_id"):
        _tt_dbg2["backtest_create"] = "SUCCESS"
        _tt_dbg2["backtest_poll"]   = "COMPLETED_NO_TRADES"
    elif _bs == "SUCCESS" and opts_result.get("backtest_id"):
        _tt_dbg2["backtest_create"] = "SUCCESS"
        _tt_dbg2["backtest_poll"]   = "COMPLETED"
    elif _bs == "BACKTEST_POLL_FAILED" and opts_result.get("backtest_id"):
        _tt_dbg2["backtest_create"] = "SUCCESS"
        _tt_dbg2["backtest_poll"]   = "FAILED"
    elif _bs == "BACKTEST_CREATE_FAILED":
        _tt_dbg2["backtest_create"] = "FAILED"
        opts_result.pop("backtest_id", None)
    # If NOT_RUN/BLOCKED: create/poll remain NOT_RUN and no backtest_id
    if not opts_result.get("backtest_id"):
        opts_result.pop("backtest_id", None)

    # ── Finalize run context provider statuses ────────────────────────────────
    if _bs in ("SUCCESS", "FLAT_NO_TRADES", "BACKTEST_POLL_FAILED", "BACKTEST_CREATE_FAILED", "ERROR"):
        _run_ctx2["provider_status"]["tastytrade"] = (
            "AUTH_OK_BACKTEST_FLAT_NO_TRADES" if _bs == "FLAT_NO_TRADES" else
            "SUCCESS" if _bs == "SUCCESS" else
            "BACKTEST_FAILED"
        )
    elif _bs in ("BLOCKED_EXTREME_STRIKE",):
        _run_ctx2["provider_status"]["tastytrade"] = "BLOCKED_EXTREME_STRIKE"
    _run_ctx2["provider_status"]["gemini"] = "SUCCESS" if ai_result.get("status") == "SUCCESS" else "FAILED"
    _run_ctx2["provider_status"]["historical_data"] = "SUCCESS"
    _final_vs = (
        "NOT_VALIDATED" if _accuracy_skip_reason else
        "SUCCESS" if _bs == "SUCCESS" else
        "BACKTEST_FAILED"
    )
    _run_ctx2["validation_status"] = _final_vs
    _run_ctx2["accuracy_eligible"]  = (opts_result.get("status") == "SUCCESS" and not _accuracy_skip_reason)
    _run_ctx2["tastytrade_debug"]    = _tt_dbg2
    st.session_state["run_context"]  = _run_ctx2

    st.session_state["opts_result"] = opts_result

    # ── Accuracy save gate ────────────────────────────────────────────────────
    # Only save when:
    # (a) backtest truly succeeded (real trades occurred)
    # (b) not exact-strike mode (since exact is unsupported — proxy result ≠ exact accuracy)
    # (c) not FLAT_NO_TRADES
    _backtest_truly_succeeded = opts_result.get("status") == "SUCCESS"
    _save_blocked_reason = _accuracy_skip_reason  # set above if exact/flat

    # Underlying stock reference — always fetch, but used for display only
    with st.spinner(f"Fetching underlying stock price on {target_date} (reference only)..."):
        val_result = run_stock_validation(spi, ai_result, hist)
    st.session_state["val_result"] = val_result

    if _backtest_truly_succeeded and not _save_blocked_reason:
        saved, save_msg = save_stock_prediction_record(spi, ai_result, val_result)
        if not val_result.get("status") == "SUCCESS":
            saved, save_msg = False, "Backtest succeeded but underlying stock validation failed — not saved"
    else:
        saved    = False
        _bt_stat = opts_result.get("status", "FAILED")
        if _save_blocked_reason:
            save_msg = f"NOT SAVED — {_save_blocked_reason}: {_bt_stat}"
        else:
            save_msg = f"NOT SAVED — options backtest {_bt_stat}"

    st.session_state["saved"]    = saved
    st.session_state["save_msg"] = save_msg


# ══════════════════════════════════════════════════════════════════════════════
# RENDER MODE 1 -- STOCK RESULTS  (O1-style side-by-side)
# ══════════════════════════════════════════════════════════════════════════════


def _render_results():
    spi      = st.session_state.get("spi", {})
    hist     = st.session_state.get("price_hist", [])
    ctx_summ = st.session_state.get("ctx_summary", {})
    ai       = st.session_state.get("ai_result", {})
    val      = st.session_state.get("val_result", {})
    hash_v   = st.session_state.get("input_hash", "")
    run_id   = st.session_state.get("run_id", "")
    run_ts   = st.session_state.get("run_ts", "")
    saved    = st.session_state.get("saved", False)
    save_msg = st.session_state.get("save_msg", "")

    symbol    = spi.get("symbol", "")
    origin    = spi.get("prediction_origin_date", "")
    target    = spi.get("target_date", "")
    horizon   = spi.get("decision_horizon_days", 30)
    ctx_start = spi.get("historical_context_start_date", "")
    cap       = float(spi.get("initial_capital", 50_000) or 50_000)

    run_type = st.session_state.get("run_type", "historical_validation")
    is_future = run_type == "live_future_prediction"

    ai_ok = ai.get("status") == "SUCCESS"
    if not ai_ok:
        _err_card(_friendly_ai_error(ai.get("error", "")))
        return

    val_ok  = bool(val and val.get("status") == "SUCCESS")
    cmp     = val.get("comparison", {}) if val_ok else {}
    cmp_ok  = cmp.get("status") == "SUCCESS"
    n_ctx   = len([b for b in hist if ctx_start <= b["date"] <= origin])
    ctx_ok  = ctx_summ.get("status") == "OK"

    # Pre-extract actual values so they are available throughout this function
    orig_p = tgt_p = ret_p = cap_v = pl_v = 0.0
    act_dec = "---"
    if val_ok:
        orig_p  = float(val.get("origin_price") or 0)
        tgt_p   = float(val.get("target_price") or 0)
        ret_p   = float(val.get("actual_return_pct") or 0)
        cap_v   = float(val.get("actual_final_capital") or cap)
        pl_v    = float(val.get("actual_total_pl") or 0)
        act_dec = val.get("actual_decision", "---")

    # ── Input Binding Warning — show if effective date differs from requested ──
    _run_ctx_stock = st.session_state.get("run_context", {})
    _ibw = _run_ctx_stock.get("input_binding_warning")
    if _ibw:
        st.markdown(
            f'<div style="background:#7C2D12;border:2px solid #DC2626;border-radius:6px;'
            f'padding:.75rem 1.1rem;margin-bottom:.5rem">'
            f'<div style="color:#FCA5A5;font-weight:900;font-size:.84rem">INPUT BINDING WARNING</div>'
            f'<div style="color:#FED7AA;font-size:.77rem;margin-top:.3rem">{_ibw}</div>'
            f'<div style="color:#FCA5A5;font-size:.72rem;margin-top:.3rem">'
            f'Your date was recorded in the snapshot. The AI used the effective date above — '
            f'NOT your requested date. See Developer Debug for full snapshot comparison.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── AI Accuracy Headline ── prominent banner at top of results ────────────
    try:
        from pathlib import Path as _Path
        import json as _json_acc
        _acc_file = _Path(__file__).resolve().parent / "stock_prediction_evaluation_runs.jsonl"
        _acc_recs: list = []
        if _acc_file.exists():
            with open(_acc_file, "r", encoding="utf-8") as _fh_acc:
                for _ln_acc in _fh_acc:
                    _ln_acc = _ln_acc.strip()
                    if _ln_acc:
                        try:
                            _acc_recs.append(_json_acc.loads(_ln_acc))
                        except Exception:
                            pass
        _acc_total = len(_acc_recs)
        _acc_sym   = str(symbol).upper()
        if _acc_total > 0:
            _acc_dec_ok  = sum(1 for r in _acc_recs if (r.get("comparison") or {}).get("decision_match") is True)
            _acc_dir_ok  = sum(1 for r in _acc_recs if (r.get("comparison") or {}).get("directional_match") is True)
            _acc_dec_pct = round(_acc_dec_ok / _acc_total * 100, 1)
            _acc_dir_pct = round(_acc_dir_ok / _acc_total * 100, 1)
            # Symbol-specific
            _sym_recs    = [r for r in _acc_recs if str((r.get("spi") or r.get("inputs") or {}).get("symbol", "")).upper() == _acc_sym]
            _sym_total   = len(_sym_recs)
            _sym_ok      = sum(1 for r in _sym_recs if (r.get("comparison") or {}).get("decision_match") is True)
            _sym_pct     = round(_sym_ok / _sym_total * 100, 1) if _sym_total else None
            # Colors
            _acc_col  = "#10B981" if _acc_dec_pct >= 70 else ("#F59E0B" if _acc_dec_pct >= 60 else "#EF4444")
            _acc_bar  = "#10B981" if _acc_dec_pct >= 70 else ("#F59E0B" if _acc_dec_pct >= 60 else "#EF4444")
            _dir_col  = "#10B981" if _acc_dir_pct >= 65 else ("#F59E0B" if _acc_dir_pct >= 50 else "#EF4444")
            _target_txt = (
                "GREAT — above 70% target" if _acc_dec_pct >= 70
                else "GOOD — above 60% minimum target" if _acc_dec_pct >= 60
                else "BELOW TARGET — aim for 60-70%+"
            )
            _sym_html = ""
            if _sym_total:
                _sym_col = "#10B981" if (_sym_pct or 0) >= 70 else ("#F59E0B" if (_sym_pct or 0) >= 60 else "#EF4444")
                _sym_html = (
                    f'<div style="text-align:center">'
                    f'<div style="color:#64748B;font-size:.65rem;text-transform:uppercase">{_acc_sym} Only</div>'
                    f'<div style="color:{_sym_col};font-size:1.4rem;font-weight:900">{_sym_pct}%</div>'
                    f'<div style="color:#475569;font-size:.62rem">{_sym_ok}/{_sym_total} runs</div>'
                    f'</div>'
                )
            st.markdown(
                f'<div style="background:#0B1A0B;border:2px solid {_acc_bar};border-radius:10px;'
                f'padding:.8rem 1.3rem;margin:.4rem 0 .6rem 0">'
                f'<div style="display:flex;align-items:center;gap:1.5rem;flex-wrap:wrap">'
                f'<div>'
                f'<div style="color:#94A3B8;font-size:.72rem;font-weight:700;text-transform:uppercase;'
                f'letter-spacing:.06em;margin-bottom:.15rem">AI Accuracy Score</div>'
                f'<div style="color:{_acc_col};font-size:2.4rem;font-weight:900;line-height:1">'
                f'{_acc_dec_pct}%</div>'
                f'<div style="color:#64748B;font-size:.68rem;margin-top:.1rem">'
                f'{_acc_dec_ok}/{_acc_total} decision matches</div>'
                f'</div>'
                f'<div style="flex:1;min-width:160px">'
                f'<div style="background:#1E293B;border-radius:4px;height:8px;margin-bottom:.4rem">'
                f'<div style="background:{_acc_col};width:{min(_acc_dec_pct,100)}%;height:8px;border-radius:4px"></div>'
                f'</div>'
                f'<div style="color:{_acc_col};font-size:.75rem;font-weight:700">{_target_txt}</div>'
                f'<div style="color:#475569;font-size:.65rem;margin-top:.2rem">'
                f'Target: 60-70% good · 70-80%+ great · Direction match: '
                f'<span style="color:{_dir_col}">{_acc_dir_pct}%</span></div>'
                f'</div>'
                f'{_sym_html}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:#0F172A;border:1px dashed #334155;border-radius:8px;'
                'padding:.6rem 1rem;margin:.3rem 0 .5rem 0;color:#475569;font-size:.77rem">'
                'AI Accuracy — <b style="color:#64748B">No historical runs saved yet.</b> '
                'Run a historical validation and the accuracy score will appear here.'
                '</div>',
                unsafe_allow_html=True,
            )
    except Exception:
        pass

    # ── Section 2 — Inputs Used + Two-Window ─────────────────────────────────
    _section_header(2, "Inputs Used", "Walk-forward setup — two-window architecture")

    _window_preview(ctx_start, origin, target, horizon)

    last_ai_bar = max((b["date"] for b in hist if b["date"] <= origin), default="---")
    i1, i2 = st.columns(2)
    with i1:
        st.markdown(
            _table([
                ("Mode",            "Stock Price Validation"),
                ("Symbol",          symbol),
                ("Benchmark",       spi.get("benchmark", "---")),
                ("Initial Capital", _fmt(cap, "$")),
                ("Price Basis",     spi.get("price_basis", "close")),
                ("Run ID",          f"<code>{run_id}</code>"),
                ("Input Hash",      f'<code style="font-size:.68rem">{hash_v}</code>'),
            ]),
            unsafe_allow_html=True,
        )
    with i2:
        st.markdown(
            _table([
                ("Context Start",     ctx_start),
                ("Prediction Origin", origin),
                ("Target Date",       target),
                ("Decision Horizon",  f"{horizon} days"),
                ("Bars to AI",        f"{n_ctx} trading days"),
                ("Last AI bar",       last_ai_bar),
                ("Total bars fetched",f"{len(hist)}"),
                ("Data Provider",     _run_ctx_stock.get("hist_provider", "---")),
            ]),
            unsafe_allow_html=True,
        )

    _eff_ctx_leak = _run_ctx_stock.get("effective_ctx_start") or ctx_start
    _leak_ctx_label = (
        f"{_eff_ctx_leak}"
        + (f' (requested {ctx_start} — provider-constrained)' if _eff_ctx_leak != ctx_start else "")
    )
    _warn_card(
        f"<b>Data Leakage Prevention:</b> AI received bars from <b>{_leak_ctx_label}</b> to "
        f"<b>{origin}</b> only. Target price ({target}) was NOT in AI context. "
        "Revealed after prediction."
    )

    # ── Execution Truth Panel ─────────────────────────────────────────────────
    _etp_ai_selected   = os.getenv("AI_PROVIDER", "gemini").upper()
    _etp_ai_used       = (ai.get("ai_provider") or "unknown").upper()
    _etp_source        = ai.get("source", "---")
    _etp_mv            = ai.get("model_version", "---")
    _etp_gemini_used   = bool(ai.get("gemini_used"))
    _etp_gemini_lat    = ai.get("gemini_latency_ms")
    _etp_gem_model     = ai.get("model_name", "---") if _etp_gemini_used else "---"
    _etp_hist_src      = get_provider_used(symbol) or "external_historical_provider"
    _etp_hist_label    = {
        "rapidapi_tradingview":         "RapidAPI TradingView",
        "external_historical_provider": "External Historical Provider (NASDAQ public)",
        "yfinance":                     "yfinance (open-source, full history)",
    }.get(_etp_hist_src, _etp_hist_src)
    _etp_tt_used       = False  # Stock mode never uses tastytrade
    _etp_rapidapi_hist = "Unavailable (subscription plan)"
    _etp_rapi_hc_etp   = st.session_state.get("rapidapi_health") or {}
    _etp_tt_hc_etp     = st.session_state.get("tastytrade_health") or {}
    _etp_rapi_called   = _etp_rapi_hc_etp.get("called", False)
    _etp_rapi_ok       = _etp_rapi_hc_etp.get("http_status") == 200
    _etp_tt_verified   = _etp_tt_hc_etp.get("customer_verified", False)
    _etp_rapidapi_used = (
        f'YES — HTTP {_etp_rapi_hc_etp.get("http_status")} ({_etp_rapi_hc_etp.get("total_count","?")}'
        f' movers)'
        if _etp_rapi_called else "NOT YET CALLED (run a prediction)"
    )
    _etp_gemini_badge  = (
        '<span style="color:#10B981;font-weight:800">YES</span>'
        if _etp_gemini_used else
        '<span style="color:#EF4444;font-weight:800">NO</span>'
    )
    _etp_gemini_key_ok = bool(
        os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
    )
    _etp_wrong_source  = (
        _etp_ai_selected == "GEMINI" and _etp_source != "gemini_stock_prediction_agent"
    )
    _etp_build_color   = "#7C3AED"
    st.markdown(
        f'<div style="background:#0A1628;border:1px solid {"#DF1B41" if _etp_wrong_source else "#7C3AED"};'
        f'border-radius:8px;padding:.7rem 1.1rem;margin:.6rem 0">'
        f'<div style="color:{"#EF4444" if _etp_wrong_source else "#A78BFA"};font-weight:800;'
        f'font-size:.82rem;margin-bottom:.5rem">EXECUTION TRUTH PANEL</div>'
        f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:.3rem .8rem;font-size:.71rem">'
        # AI block
        f'<div style="color:#93C5FD;font-weight:700">AI PROVIDER</div>'
        f'<div style="color:#93C5FD;font-weight:700">MARKET DATA</div>'
        f'<div style="color:#93C5FD;font-weight:700">VALIDATION</div>'
        # AI values
        f'<div style="color:#F1F5F9">Selected: <b>{_etp_ai_selected}</b><br>'
        f'Actually Used: <b style="color:{"#10B981" if _etp_ai_selected==_etp_ai_used else "#EF4444"}">{_etp_ai_used}</b><br>'
        f'Gemini API Called: {_etp_gemini_badge}<br>'
        f'Gemini Key Present: <b>{"YES" if _etp_gemini_key_ok else "NO"}</b><br>'
        f'Gemini Model: <b>{_etp_gem_model}</b><br>'
        f'Latency: <b>{"---" if not _etp_gemini_lat else f"{_etp_gemini_lat:.0f} ms"}</b><br>'
        f'Prediction Source: <b style="color:{"#10B981" if _etp_source=="gemini_stock_prediction_agent" else "#EF4444"}">{_etp_source}</b><br>'
        f'Model Version: <b>{_etp_mv}</b></div>'
        # Market data values
        f'<div style="color:#F1F5F9">RapidAPI Called: <b style="color:{"#10B981" if _etp_rapi_called else "#F59E0B"}">'
        f'{"YES — HTTP " + str(_etp_rapi_hc_etp.get("http_status","?")) if _etp_rapi_called else "Not yet (run prediction first)"}</b><br>'
        f'RapidAPI Status: <b>{"200 OK" if _etp_rapi_ok else ("Error" if _etp_rapi_called else "---")}</b><br>'
        f'RapidAPI OHLCV: <b style="color:#F59E0B">{_etp_rapidapi_hist}</b><br>'
        f'Historical Provider: <b style="color:#10B981">{_etp_hist_label}</b><br>'
        f'Historical Source Key: <b style="color:#10B981">{_etp_hist_src}</b></div>'
        # Validation values
        f'<div style="color:#F1F5F9">Mode: <b>Stock Price Validation</b><br>'
        f'Tastytrade Called: <b style="color:{"#10B981" if _etp_tt_verified else "#F59E0B"}">'
        f'{"YES — customer verified" if _etp_tt_verified else "Called (not verified in stock mode)"}</b><br>'
        f'Tastytrade Used: <b>NO — stock mode uses price validation</b><br>'
        f'Validation Engine: <b>historical_stock_price_validation</b></div>'
        f'</div>'
        + (
            f'<div style="background:#7F1D1D;color:#FCA5A5;padding:.4rem .7rem;'
            f'border-radius:4px;margin-top:.5rem;font-size:.72rem;font-weight:700">'
            f'WIRING ERROR: AI_PROVIDER={_etp_ai_selected} but source={_etp_source}. '
            f'Restart Streamlit: Ctrl+C then streamlit run streamlit_app.py</div>'
            if _etp_wrong_source else ""
        )
        + f'</div>',
        unsafe_allow_html=True,
    )

    # ── Paid API Usage Proof ──────────────────────────────────────────────────
    _rapi_hc_r = st.session_state.get("rapidapi_health") or {}
    _tt_hc_r   = st.session_state.get("tastytrade_health") or {}
    _rapi_called   = _rapi_hc_r.get("called", False)
    _rapi_status   = _rapi_hc_r.get("http_status", 0)
    _rapi_ok       = _rapi_called and _rapi_status == 200
    _rapi_endpoint = _rapi_hc_r.get("endpoint", "---")
    _rapi_count    = _rapi_hc_r.get("total_count")
    _rapi_syms     = _rapi_hc_r.get("top_symbols", [])
    _rapi_err      = _rapi_hc_r.get("error")
    _rapi_key_ok   = _rapi_hc_r.get("key_present", False)
    _rapi_leakage  = _rapi_hc_r.get("used_in_prediction_context", False)
    _tt_called     = _tt_hc_r.get("called", False)
    _tt_status     = _tt_hc_r.get("http_status", 0)
    _tt_ok         = _tt_hc_r.get("customer_verified", False)
    _tt_endpoint   = _tt_hc_r.get("endpoint", "---")
    _tt_refreshed  = _tt_hc_r.get("token_refreshed", False)
    _tt_err        = _tt_hc_r.get("error")
    _tt_ref_ok     = _tt_hc_r.get("refresh_present", False)

    def _api_badge(ok: bool, label_yes: str = "YES", label_no: str = "NO") -> str:
        col = "#10B981" if ok else "#EF4444"
        lbl = label_yes if ok else label_no
        return f'<span style="color:{col};font-weight:800">{lbl}</span>'

    st.markdown(
        f'<div style="background:#0A1628;border:1px solid #2563EB;'
        f'border-radius:8px;padding:.7rem 1.1rem;margin:.6rem 0">'
        f'<div style="color:#93C5FD;font-weight:800;font-size:.82rem;margin-bottom:.5rem">'
        f'PAID API USAGE PROOF</div>'
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:.3rem 1.2rem;font-size:.71rem">'
        # RapidAPI column
        f'<div style="color:#FCD34D;font-weight:700;margin-bottom:.2rem">RAPIDAPI (TradingView)</div>'
        f'<div style="color:#FCD34D;font-weight:700;margin-bottom:.2rem">TASTYTRADE (OAuth)</div>'
        f'<div style="color:#F1F5F9">'
        f'API Key Present: {_api_badge(_rapi_key_ok)}<br>'
        f'Called This Run: {_api_badge(_rapi_called)}<br>'
        f'HTTP Status: <b>{_rapi_status if _rapi_called else "---"}</b><br>'
        f'Endpoint: <code style="font-size:.66rem">{_rapi_endpoint.replace("https://","")}</code><br>'
        f'Total Movers: <b>{_rapi_count if _rapi_count is not None else "---"}</b><br>'
        f'Top Symbols: <b>{", ".join(_rapi_syms[:3]) if _rapi_syms else "---"}</b><br>'
        f'Role: <b>LIVE_MARKET_MOVERS / PROVIDER_HEALTH</b><br>'
        f'Used for Historical OHLCV: {_api_badge(False, "YES", "NO")}<br>'
        f'Used for Stock Validation: {_api_badge(False, "YES", "NO")}<br>'
        f'Used for Gemini Historical Prompt: {_api_badge(_rapi_leakage, "YES", "NO")}<br>'
        f'Data To Gemini: {_api_badge(not _rapi_leakage, "NO (leakage-safe)", "YES (LEAK!)")}<br>'
        f'Used In Stock Mode: {_api_badge(True, "YES — health check only", "NO")}<br>'
        f'Used In Options Mode: {_api_badge(True, "YES — health check only", "NO")}<br>'
        + (f'Error: <span style="color:#EF4444">{_rapi_err}</span>' if _rapi_err else "")
        + f'</div>'
        # Tastytrade column — stock mode: not used for backtesting
        f'<div style="color:#F1F5F9">'
        + (
            f'<b style="color:#EF4444">FORCE_DISABLE_TASTYTRADE=true</b><br>'
            f'Token Called: NOT_CALLED (forced disabled)<br>'
            f'Customer Check: NOT_CALLED<br>'
            f'Backtest: DISABLED<br>'
            if _tt_hc_r.get("blocked_by") == "MISSING_FORCED" else
            f'Token Checked: {_api_badge(_tt_called, "YES — health check", "NO")}<br>'
            f'Token Refreshed: {_api_badge(_tt_refreshed)}<br>'
            f'Customer Verified: {_api_badge(_tt_ok)}<br>'
            f'HTTP Status: <b>{_tt_status if _tt_called else "---"}</b><br>'
            + (f'Error: <span style="color:#EF4444">{_tt_err}</span><br>' if _tt_err else "")
        )
        + f'Role: <b>auth &amp; account health check (stock mode only)</b><br>'
        f'Used In Stock Mode: {_api_badge(False, "YES", "NO — stock uses price validation")}<br>'
        f'Used In Options Mode: {_api_badge(True, "YES — backtester", "NO")}<br>'
        f'</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Section 3 (left) | Section 4 (right) — side-by-side ──────────────────
    st.markdown(
        f'<div style="color:#8BA9C4;font-size:.72rem;margin:.8rem 0 .2rem 0">'
        f'<b>{symbol}</b> &nbsp;|&nbsp; {origin} → {target} &nbsp;({horizon} days)'
        f'</div>',
        unsafe_allow_html=True,
    )

    c3l, c3r = st.columns(2)

    with c3l:
        _ai_prov   = ai.get("ai_provider", "baseline")
        _gem_used  = bool(ai.get("gemini_used"))
        _gem_lat   = ai.get("gemini_latency_ms")
        _gem_mod   = ai.get("model_name", "---")
        _gem_label = (
            f'<span style="color:#10B981;font-weight:700">YES</span>'
            if _gem_used else
            f'<span style="color:#EF4444;font-weight:700">NO</span>'
        )
        _s3_title = "GEMINI AI STOCK PREDICTION" if _gem_used else "AI STOCK PREDICTION"
        st.markdown(
            f'<div style="background:#0F2940;padding:.4rem 1rem;border-radius:6px;'
            f'color:#93C5FD;font-weight:800;font-size:.85rem;margin-bottom:.4rem">'
            f"SECTION 3 — {_s3_title}</div>",
            unsafe_allow_html=True,
        )
        _ph = ai.get("prompt_hash", "")
        _oh = ai.get("gemini_output_hash", "")
        _dq = ai.get("data_quality_score")
        st.markdown(
            _table([
                ("Decision",              _decision_badge(ai.get("decision", "---"))),
                ("Origin Price Used",     _fmt(ai.get("origin_price_used"), "$")),
                ("Predicted Target Price",_fmt(ai.get("predicted_target_price"), "$")),
                ("Predicted Return %",    _sign_fmt(ai.get("predicted_return_pct"), suffix="%")),
                ("Predicted Final Capital",_fmt(ai.get("predicted_final_capital"), "$")),
                ("Predicted Total P&L",   _sign_fmt(ai.get("predicted_total_pl"))),
                ("Initial Capital",       _fmt(cap, "$")),
                ("Decision Horizon",      f"{horizon} days"),
                ("Confidence Score",      f"{ai.get('confidence_score', '---')}/100"),
                ("Risk Score",            f"{ai.get('risk_score', '---')}/100"),
                ("Data Quality Score",    f"{min(100, int(_dq))}/100" if _dq is not None else "---"),
                ("Effective Origin Date", ai.get("effective_origin_date", "---")),
                ("AI Provider",           f'{_ai_prov.upper()} {"(active)" if _gem_used else "(fallback baseline)"}'),
                ("Gemini Model",          _gem_mod if _gem_used else "---"),
                ("Gemini Latency",        f"{_gem_lat} ms" if _gem_used and _gem_lat else "---"),
                ("Prompt Hash",           f'<code style="font-size:.65rem">{_ph[:16]}…</code>' if _ph else "---"),
                ("Output Hash",           f'<code style="font-size:.65rem">{_oh[:16]}…</code>' if _oh else "---"),
                ("Model Version",         ai.get("model_version", "---")),
            ]),
            unsafe_allow_html=True,
        )

    with c3r:
        st.markdown(
            '<div style="background:#052E1A;padding:.4rem 1rem;border-radius:6px;'
            'color:#6EE7B7;font-weight:800;font-size:.85rem;margin-bottom:.4rem">'
            "SECTION 4 — ACTUAL HISTORICAL VALIDATION</div>",
            unsafe_allow_html=True,
        )
        if is_future:
            st.markdown(
                _table([
                    ("Status",                    "⏳ PENDING"),
                    ("Reason",                    "Target date is in the future"),
                    ("Requested Target Date",     target),
                    ("Validation Available After",target),
                    ("Actual Origin Price",       _fmt(ai.get("origin_price_used"), "$")),
                    ("Decision",                  "PENDING"),
                    ("Final Capital",             "PENDING"),
                    ("Initial Capital",           _fmt(cap, "$")),
                    ("Decision Horizon",          f"{horizon} days"),
                    ("Historical Price Provider", _etp_hist_src),
                ]),
                unsafe_allow_html=True,
            )
        elif val_ok:
            _prov = get_provider_used(spi.get("symbol", ""))
            _prov_name_map = {
                "rapidapi_tradingview": "RapidAPI TradingView",
                "yfinance":             "yfinance",
                "nasdaq_external":      "NASDAQ External",
            }
            _prov_label = _prov_name_map.get(_prov, _prov or _etp_hist_src or "Historical price data provider")
            st.markdown(
                _table([
                    ("Actual Decision",      _decision_badge(act_dec)),
                    ("Actual Origin Price",  _fmt(orig_p, "$")),
                    ("Actual Target Price",  _fmt(tgt_p, "$")),
                    ("Actual Return %",      _sign_fmt(ret_p, suffix="%")),
                    ("Actual Final Capital", _fmt(cap_v, "$")),
                    ("Actual Total P&L",     _sign_fmt(pl_v)),
                    ("Initial Capital",      _fmt(cap, "$")),
                    ("Decision Horizon",     f"{horizon} days"),
                    ("Requested Origin",     val.get("requested_prediction_origin_date", "---")),
                    ("Effective Origin",     val.get("effective_origin_price_date", "---")),
                    ("Requested Target",     val.get("requested_target_date", "---")),
                    ("Effective Target",     val.get("effective_target_price_date", "---")),
                    ("Historical Price Provider", _prov_label),
                ]),
                unsafe_allow_html=True,
            )
        else:
            _err_card(
                f"Actual validation failed: {val.get('error', 'Did not run') if val else 'Did not run'}. "
                "Possible causes: target date is a weekend/holiday, or no data for this range."
            )

    # AI reasoning — full structured breakdown
    _ai_main    = ai.get("reasoning") or ai.get("main_reason") or ""
    _ai_trend   = ai.get("trend_assessment", "") or ""
    _ai_mom     = ai.get("momentum_assessment", "") or ""
    _ai_vol     = ai.get("volatility_assessment", "") or ""
    _ai_bench   = ai.get("benchmark_assessment", "") or ""
    _ai_bull    = ai.get("bull_case", "") or ""
    _ai_bear    = ai.get("bear_case", "") or ""
    _ai_why_not = ai.get("why_not_opposite", "") or ""
    _ai_inv     = ai.get("invalidating_conditions", []) or []
    _ai_kf      = ai.get("key_features_used", []) or []
    _ai_opts    = ai.get("options_strategy_assessment", "") or ""
    if _ai_main or _ai_trend:
        _reasoning_rows = ""
        if _ai_opts:
            _reasoning_rows += (
                f'<div style="margin-bottom:.7rem;padding:.5rem .8rem;'
                f'background:#1C1009;border-left:3px solid #F59E0B;border-radius:4px">'
                f'<span style="color:#FCD34D;font-weight:700">Options Strategy Assessment:</span> {_ai_opts}</div>'
            )
        if _ai_main:
            _reasoning_rows += f'<div style="margin-bottom:.6rem"><span style="color:#93C5FD;font-weight:700">Prediction Rationale:</span> {_ai_main}</div>'
        if _ai_trend:
            _reasoning_rows += f'<div style="margin-bottom:.4rem"><span style="color:#6EE7B7;font-weight:600">Trend:</span> {_ai_trend}</div>'
        if _ai_mom:
            _reasoning_rows += f'<div style="margin-bottom:.4rem"><span style="color:#6EE7B7;font-weight:600">Momentum (RSI/MACD):</span> {_ai_mom}</div>'
        if _ai_vol:
            _reasoning_rows += f'<div style="margin-bottom:.4rem"><span style="color:#6EE7B7;font-weight:600">Volatility:</span> {_ai_vol}</div>'
        if _ai_bench:
            _reasoning_rows += f'<div style="margin-bottom:.4rem"><span style="color:#6EE7B7;font-weight:600">vs Benchmark:</span> {_ai_bench}</div>'
        if _ai_bull:
            _reasoning_rows += f'<div style="margin-bottom:.4rem"><span style="color:#10B981;font-weight:600">Bull Case:</span> {_ai_bull}</div>'
        if _ai_bear:
            _reasoning_rows += f'<div style="margin-bottom:.4rem"><span style="color:#EF4444;font-weight:600">Bear Case:</span> {_ai_bear}</div>'
        if _ai_why_not:
            _reasoning_rows += f'<div style="margin-bottom:.4rem"><span style="color:#F59E0B;font-weight:600">Why Not Opposite:</span> {_ai_why_not}</div>'
        if _ai_kf:
            _reasoning_rows += f'<div style="margin-bottom:.4rem"><span style="color:#C4B5FD;font-weight:600">Key Signals Used:</span> {", ".join(_ai_kf)}</div>'
        if _ai_inv:
            _reasoning_rows += f'<div style="margin-top:.4rem"><span style="color:#F87171;font-weight:600">Invalidating Conditions:</span> {" | ".join(_ai_inv)}</div>'
        st.markdown(
            f'<div style="background:#0A1929;border-left:4px solid #3B82F6;border-radius:6px;'
            f'padding:.9rem 1.2rem;margin:.6rem 0;font-size:.82rem;color:#CBD5E1;line-height:1.6">'
            f'{_reasoning_rows}</div>',
            unsafe_allow_html=True,
        )

    lck = ai.get("leakage_check", "")
    if lck == "CLEAN":
        fu = ai.get("features_used", {}) or {}
        _ok_card(
            f"Leakage check CLEAN — {fu.get('bars_used', '?')} bars used, "
            f"last AI bar: {fu.get('last_ai_bar_date', '?')}"
        )
    elif lck == "LEAKAGE_DETECTED":
        _err_card("LEAKAGE DETECTED in AI context bars — results are invalid.")

    # Formula proof (only when actual data is available)
    if val_ok and orig_p > 0:
        st.markdown(
            f'<div style="background:#F0FFF4;border:1px solid #6EE7B7;border-radius:6px;'
            f'padding:.6rem 1.2rem;font-size:.74rem;color:#065F46;margin:.4rem 0;font-family:monospace">'
            f"<b>Exact Formula Proof:</b><br>"
            f"actual_return_pct    = (({tgt_p:,.6f} − {orig_p:,.6f}) / {orig_p:,.6f}) × 100"
            f" = <b>{ret_p:,.6f}%</b><br>"
            f"actual_final_capital = {cap:,.2f} × ({tgt_p:,.6f} / {orig_p:,.6f})"
            f" = <b>${cap_v:,.6f}</b>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Numeric Closeness ────────────────────────────────────────────────────
    _section_header(None, "Numeric Closeness", "AI prediction error vs actual")

    if cmp_ok:
        dec_match   = cmp.get("decision_match", False)
        dir_match   = cmp.get("directional_match", False)
        ret_err     = float(cmp.get("return_error_pct") or 0)
        cap_err     = float(cmp.get("capital_error") or 0)
        cap_err_pct = cmp.get("capital_error_pct")
        price_err   = float(cmp.get("target_price_error") or 0)
        pl_err      = float(cmp.get("pl_error") or 0)

        _dm_col   = "#059669" if dec_match  else "#DC2626"
        _dir_col  = "#059669" if dir_match  else "#DC2626"
        _pe_col   = "#DC2626" if abs(price_err) > 20 else ("#F59E0B" if abs(price_err) > 5 else "#059669")
        _re_col   = "#DC2626" if abs(ret_err) > 20   else ("#F59E0B" if abs(ret_err) > 5 else "#059669")
        _ce_col   = "#DC2626" if abs(cap_err) > 5000 else ("#F59E0B" if abs(cap_err) > 1000 else "#059669")
        _pl_col   = "#DC2626" if abs(pl_err) > 5000  else ("#F59E0B" if abs(pl_err) > 1000 else "#059669")
        st.markdown(
            f'<div style="display:flex;gap:.6rem;flex-wrap:wrap;margin:.4rem 0">'
            f'<div style="flex:1;min-width:100px;background:#F8FAFC;border:1px solid #E5E7EB;'
            f'border-radius:6px;padding:.5rem .7rem;text-align:center">'
            f'<div style="font-size:.63rem;color:#6B7280;font-weight:600">Decision Match</div>'
            f'<div style="font-size:1rem;font-weight:800;color:{_dm_col}">{"YES" if dec_match else "NO"}</div></div>'
            f'<div style="flex:1;min-width:100px;background:#F8FAFC;border:1px solid #E5E7EB;'
            f'border-radius:6px;padding:.5rem .7rem;text-align:center">'
            f'<div style="font-size:.63rem;color:#6B7280;font-weight:600">Direction Match</div>'
            f'<div style="font-size:1rem;font-weight:800;color:{_dir_col}">{"YES" if dir_match else "NO"}</div></div>'
            f'<div style="flex:1;min-width:100px;background:#F8FAFC;border:1px solid #E5E7EB;'
            f'border-radius:6px;padding:.5rem .7rem;text-align:center">'
            f'<div style="font-size:.63rem;color:#6B7280;font-weight:600">Price Error</div>'
            f'<div style="font-size:1rem;font-weight:800;color:{_pe_col}">${price_err:+,.2f}</div></div>'
            f'<div style="flex:1;min-width:100px;background:#F8FAFC;border:1px solid #E5E7EB;'
            f'border-radius:6px;padding:.5rem .7rem;text-align:center">'
            f'<div style="font-size:.63rem;color:#6B7280;font-weight:600">Return Error</div>'
            f'<div style="font-size:1rem;font-weight:800;color:{_re_col}">{ret_err:+.2f}pp</div></div>'
            f'<div style="flex:1;min-width:100px;background:#F8FAFC;border:1px solid #E5E7EB;'
            f'border-radius:6px;padding:.5rem .7rem;text-align:center">'
            f'<div style="font-size:.63rem;color:#6B7280;font-weight:600">Capital Error</div>'
            f'<div style="font-size:1rem;font-weight:800;color:{_ce_col}">${cap_err:+,.0f}</div></div>'
            f'<div style="flex:1;min-width:100px;background:#F8FAFC;border:1px solid #E5E7EB;'
            f'border-radius:6px;padding:.5rem .7rem;text-align:center">'
            f'<div style="font-size:.63rem;color:#6B7280;font-weight:600">P&L Error</div>'
            f'<div style="font-size:1rem;font-weight:800;color:{_pl_col}">${pl_err:+,.0f}</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        c4a, c4b, c4c = st.columns([5, 5, 4])
        with c4a:
            st.markdown("**AI PREDICTION**")
            st.markdown(
                _table([
                    ("Decision",      _decision_badge(cmp.get("ai_decision", "---"))),
                    ("Target Price",  _fmt(cmp.get("ai_predicted_price"), "$")),
                    ("Return %",      _fmt(cmp.get("ai_predicted_return_pct"), suffix="%")),
                    ("Final Capital", _fmt(cmp.get("ai_predicted_capital"), "$")),
                    ("Total P&L",     _sign_fmt(ai.get("predicted_total_pl"))),
                ]),
                unsafe_allow_html=True,
            )
        with c4b:
            st.markdown("**ACTUAL HISTORICAL (GROUND TRUTH)**")
            st.markdown(
                _table([
                    ("Decision",     _decision_badge(cmp.get("actual_decision", "---"))),
                    ("Target Price", _fmt(cmp.get("actual_price"), "$")),
                    ("Return %",     _fmt(cmp.get("actual_return_pct"), suffix="%")),
                    ("Final Capital",_fmt(cmp.get("actual_capital"), "$")),
                    ("Total P&L",    _sign_fmt(pl_v)),
                ]),
                unsafe_allow_html=True,
            )
        with c4c:
            st.markdown("**ERROR (AI minus Actual)**")
            st.markdown(
                _table([
                    ("Decision Match",  "YES" if dec_match else "NO"),
                    ("Direction Match", "YES" if dir_match else "NO"),
                    ("Price Error $",   _err_color(price_err)),
                    ("Return Error pp", _err_color(ret_err, "pp")),
                    ("Capital Error $", _err_color(cap_err)),
                    ("Capital Error %", _err_color(cap_err_pct, "%") if cap_err_pct is not None else "---"),
                    ("P&L Error $",     _err_color(pl_err)),
                ]),
                unsafe_allow_html=True,
            )
    else:
        _warn_card("Numeric closeness unavailable — actual validation did not succeed.")

    # ── Final Decision Board ─────────────────────────────────────────────────
    _section_header(None, "Final Decision Board")

    ai_dec = ai.get("decision", "---")
    if is_future:
        agreement = "PENDING"
        final_dec = "UNVERIFIED"
        act_dec_display = "PENDING"
    else:
        agreement       = cmp.get("agreement", "UNVERIFIED") if cmp_ok else "UNVERIFIED"
        final_dec       = cmp.get("final_decision", "REVIEW") if cmp_ok else "REVIEW"
        act_dec_display = act_dec

    c5a, c5b, c5c, c5d = st.columns(4)
    with c5a:
        st.markdown(
            '<div style="background:#0A2540;padding:.6rem 1rem;border-radius:6px;text-align:center">'
            '<div style="color:#8BA9C4;font-size:.7rem;margin-bottom:.3rem">AI PREDICTED DECISION</div>'
            f'<div style="font-size:1.6rem;font-weight:900">{_decision_badge(ai_dec)}</div>'
            f'<div style="color:#8BA9C4;font-size:.66rem;margin-top:.2rem">'
            f"Return: {_sign_fmt(ai.get('predicted_return_pct'), suffix='%')}"
            "</div></div>",
            unsafe_allow_html=True,
        )
    with c5b:
        _act_badge = (
            '<span style="color:#D97706;font-weight:900">⏳ PENDING</span>'
            if is_future else _decision_badge(act_dec_display)
        )
        st.markdown(
            '<div style="background:#0A2540;padding:.6rem 1rem;border-radius:6px;text-align:center">'
            '<div style="color:#8BA9C4;font-size:.7rem;margin-bottom:.3rem">ACTUAL DECISION</div>'
            f'<div style="font-size:1.4rem;font-weight:900">{_act_badge}</div>'
            f'<div style="color:#8BA9C4;font-size:.66rem;margin-top:.2rem">'
            f"{'Awaiting target date' if is_future else _sign_fmt(ret_p if val_ok else None, suffix='%')}"
            "</div></div>",
            unsafe_allow_html=True,
        )
    with c5c:
        agree_color = (
            "#D97706" if is_future else
            ("#00875A" if agreement == "MATCH" else ("#DF1B41" if agreement == "CONFLICT" else "#D97706"))
        )
        agree_note = (
            "Target date pending" if is_future else
            ("Decision match" if agreement == "MATCH" else ("AI was wrong" if agreement == "CONFLICT" else "Unverified"))
        )
        st.markdown(
            '<div style="background:#0A2540;padding:.6rem 1rem;border-radius:6px;text-align:center">'
            '<div style="color:#8BA9C4;font-size:.7rem;margin-bottom:.3rem">AGREEMENT</div>'
            f'<div style="font-size:1.4rem;font-weight:900;color:{agree_color}">{agreement}</div>'
            f'<div style="color:#8BA9C4;font-size:.66rem;margin-top:.2rem">{agree_note}</div>'
            "</div>",
            unsafe_allow_html=True,
        )
    with c5d:
        st.markdown(
            '<div style="background:#0A2540;padding:.6rem 1rem;border-radius:6px;text-align:center">'
            '<div style="color:#8BA9C4;font-size:.7rem;margin-bottom:.3rem">FINAL VERIFIED DECISION</div>'
            f'<div style="font-size:1.6rem;font-weight:900">{_decision_badge(final_dec)}</div>'
            f'<div style="color:#8BA9C4;font-size:.66rem;margin-top:.2rem">'
            f"{'Awaiting validation' if is_future else ('Conflict → human review' if agreement == 'CONFLICT' else ('Both engines agree' if agreement == 'MATCH' else 'Review needed'))}"
            "</div></div>",
            unsafe_allow_html=True,
        )

    if is_future:
        st.info(
            f"LIVE FUTURE PREDICTION — AI predicts {ai_dec} "
            f"({_sign_fmt(ai.get('predicted_return_pct'), suffix='%')} in {horizon} days). "
            f"Actual validation available after {target}. Pending record saved."
        )
    elif agreement == "MATCH":
        _qc = cmp.get("quality_classification", "") if cmp_ok else ""
        _ret_err_abs = abs(float(cmp.get("return_error_pct") or 0)) if cmp_ok else 0
        if _qc == "SEVERE_MAGNITUDE_FAILURE":
            st.warning(
                f"DIRECTION MATCH, BUT SEVERE MAGNITUDE ERROR — "
                f"AI predicted {ai_dec} (return +{abs(float(ai.get('predicted_return_pct') or 0)):.2f}%), "
                f"actual was {act_dec} (return {_sign_fmt(ret_p if val_ok else None, suffix='%')}). "
                f"Return error: {_ret_err_abs:.2f}pp. "
                "Decision direction correct but magnitude was severely wrong. Use this as calibration failure."
            )
        elif _qc in ("MAGNITUDE_FAILURE", "LOW_QUALITY_MATCH"):
            st.success(
                f"MATCH — AI predicted {ai_dec}, actual was {act_dec}. "
                + (f"Return error: {_ret_err_abs:.2f}pp — {_qc.replace('_', ' ')}. Calibration improvement recommended." if cmp_ok else "")
            )
        else:
            st.success(
                f"MATCH — AI predicted {ai_dec}, actual was {act_dec}. "
                + (f"Capital error: ${abs(float(cmp.get('capital_error') or 0)):,.2f}. "
                   f"Return error: {_ret_err_abs:.2f}pp." if cmp_ok else "")
            )
        if _qc:
            st.markdown(
                f'<div style="background:#FEF9C3;border:1px solid #F59E0B;border-radius:5px;'
                f'padding:.4rem .8rem;font-size:.74rem;color:#78350F;margin:.3rem 0">'
                f'<b>Numeric Quality:</b> {_qc}'
                + (' — NOT counted as clean match in accuracy headline' if _qc not in ('CLEAN_MATCH', 'DIRECTION_MATCH_MINOR_ERROR') else ' — counted as clean match')
                + '</div>', unsafe_allow_html=True
            )
    elif agreement == "CONFLICT":
        st.error(
            f"CONFLICT — AI predicted {ai_dec} but actual was {act_dec}. "
            "Conflict record saved for accuracy metrics. Final decision: REVIEW."
        )
    else:
        st.warning(f"{agreement} — {cmp.get('agreement_note', 'Actual validation incomplete.')}")

    if saved:
        st.success(f"Record saved to accuracy log. {save_msg}")
    else:
        st.info(f"Record not saved: {save_msg}")

    # ── Benchmark missing warning ─────────────────────────────────────────────
    _bench_sym = spi.get("benchmark", "")
    if _bench_sym:
        _fp_bm = (ai.get("_feature_packet") or {}).get("benchmark_comparison") or {}
        _bm_vals = [
            _fp_bm.get("benchmark_return_20d"),
            _fp_bm.get("benchmark_return_60d"),
            _fp_bm.get("relative_strength_20d"),
            _fp_bm.get("relative_strength_60d"),
        ]
        if all(v is None for v in _bm_vals):
            _warn_card(
                f"<b>Benchmark {_bench_sym} selected but benchmark comparison data was unavailable.</b><br>"
                "All benchmark_comparison fields (benchmark_return_20d, benchmark_return_60d, "
                "relative_strength_20d, relative_strength_60d) are NULL in the feature packet. "
                "AI did not have benchmark context for this run."
            )

    # ── Section 8 — Known Answer Audit ───────────────────────────────────────
    _render_known_answer_audit(val, cap)

    # ── Developer Debug (collapsed expanders below) ──────────────────────────
    fu  = ai.get("features_used", {}) or {}
    mom = ai.get("_momentum_signals", {}) or {}
    _gem_used_d  = bool(ai.get("gemini_used"))
    _ai_prov_d   = ai.get("ai_provider", "baseline")
    _fp_d        = ai.get("_feature_packet", {}) or {}
    _cal_d       = ai.get("_calibration_summary", {}) or {}
    _gem_raw_d   = ai.get("_gemini_raw_json", {}) or {}

    # ── User Input Snapshot vs Effective Input ──────────────────────────────
    _snap = st.session_state.get("user_input_snapshot") or {}
    _eff  = {
        "effective_origin_date":    origin,
        "effective_ctx_start_date": ctx_start,
        "price_basis":              ai.get("price_basis", "close"),
        "bars_given_to_ai":         fu.get("bars_used", "?"),
        "last_ai_bar_date":         fu.get("last_ai_bar_date", "?"),
        "leakage_check":            ai.get("leakage_check", "pass"),
    }
    with st.expander("User Input Snapshot vs Effective Input", expanded=False):
        _c_snap, _c_eff = st.columns(2)
        with _c_snap:
            st.caption("User Requested")
            st.json({k: str(v) for k, v in _snap.items()} if _snap else {"status": "not available"})
        with _c_eff:
            st.caption("System Effective")
            st.json({k: str(v) for k, v in _eff.items()})
        st.caption(
            "target-date price was NOT in these bars — the AI only saw bars up to the origin date. "
            "Target price is hidden from AI context. This is the walk-forward leakage guard."
        )

    c9a, c9b = st.columns(2)
    with c9a:
        st.markdown(
            _table([
                ("Run ID",                f"<code>{run_id}</code>"),
                ("Run Timestamp (UTC)",   run_ts),
                ("Input Hash",            f'<code style="font-size:.68rem">{hash_v}</code>'),
                ("AI Output Hash",        f'<code style="font-size:.68rem">{ai.get("gemini_output_hash", "---")}</code>'),
                ("Hash Match",            "YES" if ai.get("stock_prediction_input_hash") == hash_v else "NO"),
                ("Leakage Check",              ai.get("leakage_check", "---")),
                ("Bars visible to AI",         f'{fu.get("bars_used", "---")} (cutoff {origin})'),
                ("Last AI bar date",           fu.get("last_ai_bar_date", "---")),
                ("Target price hidden",        "YES — not in AI context"),
                ("actual_provider_used",       _etp_hist_src),
                ("actual_validation_engine",   "historical_stock_price_validation"),
                ("ai_source",                  ai.get("source", "---")),
                ("Model version",              ai.get("model_version", "---")),
            ]),
            unsafe_allow_html=True,
        )
    with c9b:
        _gem_label2 = (
            f'<span style="color:#10B981;font-weight:700">YES</span>'
            if _gem_used_d else
            f'<span style="color:#EF4444;font-weight:700">NO</span>'
        )
        st.markdown(
            _table([
                ("AI Provider",       _ai_prov_d.upper()),
                ("Gemini Used",       _gem_label2),
                ("Gemini Model",      ai.get("model_name", "---") if _gem_used_d else "---"),
                ("Gemini Latency",    f'{ai.get("gemini_latency_ms","---")} ms' if _gem_used_d else "---"),
                ("Prompt Hash",       f'<code style="font-size:.68rem">{ai.get("prompt_hash","---")}</code>' if _gem_used_d else "---"),
                ("Gemini Out Hash",   f'<code style="font-size:.68rem">{ai.get("gemini_output_hash","---")}</code>' if _gem_used_d else "---"),
                ("1w momentum (5d)",  _fmt(fu.get("return_5d"), suffix="%")),
                ("1mo momentum (20d)",_fmt(fu.get("return_20d"), suffix="%")),
                ("3mo momentum (60d)",_fmt(fu.get("return_60d"), suffix="%")),
                ("RSI_14",            str(fu.get("RSI_14", "---"))),
                ("Trend regime",      str(fu.get("trend_regime", "---"))),
                ("Reversal risk",     str(fu.get("reversal_risk", "---"))),
                ("Annualized vol",    _fmt(fu.get("annualized_vol"), suffix="%")),
            ]),
            unsafe_allow_html=True,
        )

    import json as _json_mod
    with st.expander("Gemini Raw Response JSON", expanded=False):
        if _gem_raw_d:
            st.json(_gem_raw_d)
        else:
            st.caption("No Gemini raw response captured for this run (baseline provider used or not yet available).")

    with st.expander("Gemini Feature Packet JSON", expanded=False):
        if _fp_d:
            st.json(_fp_d)
        else:
            st.caption("No feature packet captured for this run.")

    _render_decision_distribution_diagnostics()


# ══════════════════════════════════════════════════════════════════════════════
# RENDER MODE 2 -- OPTIONS RESULTS  (O1-style side-by-side, O2-correct dates)
# ══════════════════════════════════════════════════════════════════════════════


def _render_options_results():
    spi             = st.session_state.get("spi", {})
    hist            = st.session_state.get("price_hist", [])
    ctx_summ        = st.session_state.get("ctx_summary", {})
    ai              = st.session_state.get("ai_result", {})
    val             = st.session_state.get("val_result", {})
    opts            = st.session_state.get("opts_result", {})
    opts_p          = st.session_state.get("opts_params", {})
    bt_payload      = st.session_state.get("backtest_payload", {})
    hash_v          = st.session_state.get("input_hash", "")
    run_id          = st.session_state.get("run_id", "")
    run_ts          = st.session_state.get("run_ts", "")
    saved           = st.session_state.get("saved", False)
    save_msg        = st.session_state.get("save_msg", "")

    symbol    = spi.get("symbol", "")
    origin    = spi.get("prediction_origin_date", "")
    target    = spi.get("target_date", "")
    horizon   = spi.get("decision_horizon_days", 30)
    ctx_start = spi.get("historical_context_start_date", "")
    cap       = float(spi.get("initial_capital", 50_000) or 50_000)

    ai_ok = ai.get("status") == "SUCCESS"
    if not ai_ok:
        _err_card(_friendly_ai_error(ai.get("error", "")))
        return

    opts_status = opts.get("status", "SKIPPED")
    val_ok      = bool(val and val.get("status") == "SUCCESS")
    cmp         = val.get("comparison", {}) if val_ok else {}
    cmp_ok      = cmp.get("status") == "SUCCESS"

    # ── Input Binding Warning — show if effective date differs from requested ──
    _run_ctx_opts = st.session_state.get("run_context", {})
    _ibw_opts = _run_ctx_opts.get("input_binding_warning")
    if _ibw_opts:
        st.markdown(
            f'<div style="background:#7C2D12;border:2px solid #DC2626;border-radius:6px;'
            f'padding:.75rem 1.1rem;margin-bottom:.5rem">'
            f'<div style="color:#FCA5A5;font-weight:900;font-size:.84rem">INPUT BINDING WARNING</div>'
            f'<div style="color:#FED7AA;font-size:.77rem;margin-top:.3rem">{_ibw_opts}</div>'
            f'<div style="color:#FCA5A5;font-size:.72rem;margin-top:.3rem">'
            f'Your date was recorded in the snapshot. The AI used the effective date above — '
            f'NOT your requested date. See Developer Debug for full snapshot comparison.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── AI Accuracy Headline ── prominent banner at top of options results ──────
    try:
        from pathlib import Path as _OPath
        import json as _ojson
        _oacc_file = _OPath(__file__).resolve().parent / "stock_prediction_evaluation_runs.jsonl"
        _oacc_recs: list = []
        if _oacc_file.exists():
            with open(_oacc_file, "r", encoding="utf-8") as _ofh:
                for _oln in _ofh:
                    _oln = _oln.strip()
                    if _oln:
                        try:
                            _oacc_recs.append(_ojson.loads(_oln))
                        except Exception:
                            pass
        _oacc_total = len(_oacc_recs)
        _oacc_sym   = str(symbol).upper()
        if _oacc_total > 0:
            _oacc_dec_ok  = sum(1 for r in _oacc_recs if (r.get("comparison") or {}).get("decision_match") is True)
            _oacc_dir_ok  = sum(1 for r in _oacc_recs if (r.get("comparison") or {}).get("directional_match") is True)
            _oacc_dec_pct = round(_oacc_dec_ok / _oacc_total * 100, 1)
            _oacc_dir_pct = round(_oacc_dir_ok / _oacc_total * 100, 1)
            _osym_recs    = [r for r in _oacc_recs if str((r.get("spi") or r.get("inputs") or {}).get("symbol", "")).upper() == _oacc_sym]
            _osym_total   = len(_osym_recs)
            _osym_ok      = sum(1 for r in _osym_recs if (r.get("comparison") or {}).get("decision_match") is True)
            _osym_pct     = round(_osym_ok / _osym_total * 100, 1) if _osym_total else None
            _oacc_col     = "#10B981" if _oacc_dec_pct >= 70 else ("#F59E0B" if _oacc_dec_pct >= 60 else "#EF4444")
            _odir_col     = "#10B981" if _oacc_dir_pct >= 65 else ("#F59E0B" if _oacc_dir_pct >= 50 else "#EF4444")
            _otarget_txt  = (
                "GREAT — above 70% target" if _oacc_dec_pct >= 70
                else "GOOD — above 60% minimum" if _oacc_dec_pct >= 60
                else "BELOW TARGET — aim for 60-70%+"
            )
            _osym_html = ""
            if _osym_total:
                _osym_col = "#10B981" if (_osym_pct or 0) >= 70 else ("#F59E0B" if (_osym_pct or 0) >= 60 else "#EF4444")
                _osym_html = (
                    f'<div style="text-align:center">'
                    f'<div style="color:#64748B;font-size:.65rem;text-transform:uppercase">{_oacc_sym} Only</div>'
                    f'<div style="color:{_osym_col};font-size:1.4rem;font-weight:900">{_osym_pct}%</div>'
                    f'<div style="color:#475569;font-size:.62rem">{_osym_ok}/{_osym_total} runs</div>'
                    f'</div>'
                )
            st.markdown(
                f'<div style="background:#0B1A0B;border:2px solid {_oacc_col};border-radius:10px;'
                f'padding:.8rem 1.3rem;margin:.4rem 0 .6rem 0">'
                f'<div style="display:flex;align-items:center;gap:1.5rem;flex-wrap:wrap">'
                f'<div>'
                f'<div style="color:#94A3B8;font-size:.72rem;font-weight:700;text-transform:uppercase;'
                f'letter-spacing:.06em;margin-bottom:.15rem">AI Accuracy Score</div>'
                f'<div style="color:{_oacc_col};font-size:2.4rem;font-weight:900;line-height:1">'
                f'{_oacc_dec_pct}%</div>'
                f'<div style="color:#64748B;font-size:.68rem;margin-top:.1rem">'
                f'{_oacc_dec_ok}/{_oacc_total} decision matches</div>'
                f'</div>'
                f'<div style="flex:1;min-width:160px">'
                f'<div style="background:#1E293B;border-radius:4px;height:8px;margin-bottom:.4rem">'
                f'<div style="background:{_oacc_col};width:{min(_oacc_dec_pct,100)}%;height:8px;border-radius:4px"></div>'
                f'</div>'
                f'<div style="color:{_oacc_col};font-size:.75rem;font-weight:700">{_otarget_txt}</div>'
                f'<div style="color:#475569;font-size:.65rem;margin-top:.2rem">'
                f'Target: 60-70% good · 70-80%+ great · Direction match: '
                f'<span style="color:{_odir_col}">{_oacc_dir_pct}%</span></div>'
                f'</div>'
                f'{_osym_html}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:#0F172A;border:1px dashed #334155;border-radius:8px;'
                'padding:.6rem 1rem;margin:.3rem 0 .5rem 0;color:#475569;font-size:.77rem">'
                'AI Accuracy — <b style="color:#64748B">No historical runs saved yet.</b> '
                'Run a historical validation on either page to populate the accuracy score.'
                '</div>',
                unsafe_allow_html=True,
            )
    except Exception:
        pass

    # ── Section 2 — Historical Context ───────────────────────────────────────
    _eff_ctx  = _run_ctx_opts.get("effective_ctx_start") or ctx_start
    _req_ctx  = _run_ctx_opts.get("requested_ctx_start") or ctx_start
    _ctx_constrained = (_eff_ctx and _req_ctx and _eff_ctx != _req_ctx)
    _ctx_display_start = _eff_ctx or ctx_start

    n_ctx  = len([b for b in hist if _ctx_display_start <= b["date"] <= origin])
    ctx_ok = ctx_summ.get("status") == "OK"

    # Show key origin price prominently, collapse the rest into expander
    if ctx_ok:
        _cc1, _cc2, _cc3 = st.columns(3)
        _cc1.metric("Origin Price (AI's last close)", f"${ctx_summ.get('end_price', 0):,.2f}")
        _cc2.metric("Context Bars Passed to AI", f"{n_ctx} trading days")
        _cc3.metric("Context Period Return", f"{ctx_summ.get('return_pct', 0):+.2f}%")
        if _ctx_constrained:
            st.caption(f"Provider-constrained: requested {_req_ctx}, effective {_eff_ctx}.")

    with st.expander("Historical Context Detail", expanded=False):
        _ctx_window_label = (
            f"{_eff_ctx}  to  {origin}"
            + (f'  <span style="color:#D97706;font-size:.72rem">(requested {_req_ctx} — provider-constrained)</span>'
               if _ctx_constrained else "")
        )
        st.markdown(
            _table([
                ("Symbol",                    symbol),
                ("Historical context window", _ctx_window_label),
                ("Prediction / validation window", f"{origin}  to  {target}  ({horizon} days)"),
                ("Last bar visible to AI",     max((b["date"] for b in hist if b["date"] <= origin), default="---")),
                ("Bars passed to AI",          f"{n_ctx} trading days"),
                ("Context start price",        _fmt(ctx_summ.get("start_price"), "$") if ctx_ok else "---"),
                ("Origin price (AI's last close)", _fmt(ctx_summ.get("end_price"), "$") if ctx_ok else "---"),
                ("Context period return",      _fmt(ctx_summ.get("return_pct"), suffix="%") if ctx_ok else "---"),
            ]),
            unsafe_allow_html=True,
        )

    # ── Section 2B — Inputs Summary (collapsed) ──────────────────────────────
    with st.expander("Run Inputs & Parameters", expanded=False):
        i1, i2 = st.columns(2)
        with i1:
            st.markdown(
                _table([
                    ("Symbol",              symbol),
                    ("Context Start",       f"{_eff_ctx}" + (f" (requested: {_req_ctx})" if _ctx_constrained else "")),
                    ("Prediction Origin",   origin),
                    ("Target Date",         target),
                    ("Capital",             _fmt(cap, "$")),
                    ("Benchmark",           spi.get("benchmark", "---")),
                    ("Price Basis",         spi.get("price_basis", "close")),
                    ("Data Provider",       _run_ctx_opts.get("hist_provider", "---")),
                    ("Mode",                "Options Strategy Validation"),
                    ("Input Hash",          f'<code style="font-size:.68rem">{hash_v}</code>'),
                ]),
                unsafe_allow_html=True,
            )
        with i2:
            _p_sel = opts_p.get("strike_selection", "delta")
            _p_is_strike = _p_sel == "strike"
            _p_strike_row = (
                ("Strike",  str(opts_p.get("strike_price", "---")))
                if _p_is_strike else
                ("Delta",   str(opts_p.get("delta", "---")))
            )
            _p_expiry_row = (
                [("Expiry Date", opts_p.get("expiry_date") or "---")]
                if _p_is_strike else []
            )
            st.markdown(
                _table([
                    ("Direction",          opts_p.get("direction", "---")),
                    ("Type",               opts_p.get("opt_type", "---")),
                    ("Quantity",           f"{opts_p.get('quantity', '---')} contract(s)"),
                    ("Strike Selection",   _p_sel.title()),
                    _p_strike_row,
                    ("DTE",                f"{opts_p.get('dte', '---')} days"),]
                    + _p_expiry_row + [
                    ("Entry Schedule",     opts_p.get("entry_schedule", "---")),
                    ("Exit: Take Profit",  f"{opts_p.get('take_profit_pct')}% of premium" if opts_p.get("take_profit_pct") else "---"),
                    ("Exit: Stop Loss",    f"{opts_p.get('stop_loss_pct')}% of premium"   if opts_p.get("stop_loss_pct")   else "---"),
                    ("Exit: After N Days", f"{opts_p.get('exit_after_days')} days"         if opts_p.get("exit_after_days") else "---"),
                    ("Backtest Window",    opts.get("backtest_range") or f"{ctx_start}  →  {target}"),
                ]),
                unsafe_allow_html=True,
            )

    # ── Paid API Usage Proof (options mode) — single truth source ────────────
    # Collapsed by default — technical debugging info, not for presentation
    _rapi_hc_o   = st.session_state.get("rapidapi_health") or {}
    _run_ctx_tt  = st.session_state.get("run_context", {})
    # Single auth truth object — same one used by Developer Debug
    _tt_truth_ui = _run_ctx_tt.get("tastytrade_auth_truth", {})
    _current_rid = _run_ctx_tt.get("run_id", "")

    _ro_called   = _rapi_hc_o.get("called", False)
    _ro_status   = _rapi_hc_o.get("http_status", 0)
    _ro_endpoint = _rapi_hc_o.get("endpoint", "---")
    _ro_count    = _rapi_hc_o.get("total_count")
    _ro_syms     = _rapi_hc_o.get("top_symbols", [])
    _ro_err      = _rapi_hc_o.get("error")
    _ro_key_ok   = _rapi_hc_o.get("key_present", False)
    _ro_leakage  = _rapi_hc_o.get("used_in_prediction_context", False)

    # Tastytrade truth — read from auth_truth, never from stale tastytrade_health
    _tt_cred_src   = _tt_truth_ui.get("credential_source", "NOT_RUN")
    _tt_rt_present = _tt_truth_ui.get("refresh_token_present", False)
    _tt_at_present = _tt_truth_ui.get("access_token_present", False)
    _tt_ref_att    = _tt_truth_ui.get("token_refresh_attempted", False)
    _tt_ref_stat   = _tt_truth_ui.get("token_refresh_status", "NOT_RUN")
    _tt_cust_stat  = _tt_truth_ui.get("customer_check_status", "NOT_RUN")
    _tt_http       = _tt_truth_ui.get("auth_http_status")
    _tt_allowed_ui = _tt_truth_ui.get("backtest_allowed", False)
    _tt_reason_ui  = _tt_truth_ui.get("reason", "---")

    def _ab(ok: bool, y: str = "YES", n: str = "NO") -> str:
        c = "#10B981" if ok else "#EF4444"
        return f'<span style="color:{c};font-weight:800">{y if ok else n}</span>'

    def _stat_badge(s: str) -> str:
        c = "#10B981" if s == "SUCCESS" else "#F59E0B" if s == "NOT_RUN" else "#EF4444"
        return f'<span style="color:{c};font-weight:700">{s}</span>'

    # Pull historical provider chain from run_context
    _hist_prov_used  = _run_ctx_tt.get("hist_provider", "unknown")
    _hist_prov_chain = _run_ctx_tt.get("provider_chain", [])
    _prov_label_map  = {
        "rapidapi_tradingview": "TradingView RapidAPI (your paid subscription)",
        "nasdaq_external":      "NASDAQ Official API (api.nasdaq.com — real exchange data)",
        "yfinance":             "yfinance (last resort fallback)",
    }
    _prov_used_label = _prov_label_map.get(_hist_prov_used, _hist_prov_used)

    def _chain_row(p: dict) -> str:
        pname  = _prov_label_map.get(p.get("provider",""), p.get("provider",""))
        bars   = p.get("bars", 0)
        status = p.get("status", "FAILED")
        err    = p.get("error", "") or ""
        is_used = p.get("provider","") == _hist_prov_used
        tick   = "✅ USED" if (is_used and bars > 0) else ("⚠️ TRIED" if bars == 0 else "✅")
        color  = "#10B981" if (is_used and bars > 0) else "#F59E0B" if bars == 0 else "#10B981"
        err_snip = f' — <span style="color:#FCA5A5">{err[:80]}</span>' if err and bars == 0 else ""
        return (f'<span style="color:{color};font-weight:700">{tick}</span> '
                f'<b>{pname}</b> → {bars} bars{err_snip}<br>')

    with st.expander("API Usage Proof & Auth Details", expanded=False):
        # ── Historical OHLCV provider chain (always shown first — most important) ──
        _chain_html = "".join(_chain_row(p) for p in _hist_prov_chain) if _hist_prov_chain else (
            f'<b>{_prov_used_label}</b> (provider chain not recorded for this run)<br>'
        )
        st.markdown(
            f'<div style="background:#0A1628;border:1px solid #2563EB;'
            f'border-radius:8px;padding:.7rem 1.1rem;margin:.6rem 0;font-size:.72rem">'
            f'<div style="color:#93C5FD;font-weight:800;font-size:.82rem;margin-bottom:.4rem">'
            f'PAID API USAGE PROOF — Run {_current_rid or "N/A"}</div>'
            # Historical OHLCV section
            f'<div style="color:#FCD34D;font-weight:700;margin-bottom:.3rem">── HISTORICAL OHLCV DATA (for AI training) ──</div>'
            f'<div style="color:#F1F5F9;margin-bottom:.5rem">'
            f'<b>Tried in order:</b> TradingView RapidAPI → NASDAQ Official API → yfinance<br>'
            f'{_chain_html}'
            f'<b>Final provider used:</b> <span style="color:#10B981;font-weight:700">{_prov_used_label}</span><br>'
            + (
                f'<span style="color:#EF4444;font-weight:700;font-size:.72rem">'
                f'⚠ WARNING: yfinance IS being used as last resort — both TradingView RapidAPI AND NASDAQ Official API '
                f'returned 0 bars for this symbol/date range. yfinance data may differ from TastyTrade prices. '
                f'Results should be treated as approximate.</span>'
                if _hist_prov_used == "yfinance" else
                f'<span style="color:#93C5FD;font-size:.68rem">'
                f'Note: TradingView RapidAPI is tried first (your paid subscription). '
                f'If it returns 0 bars for the requested date range, the system automatically falls back to '
                f'NASDAQ\'s official public API (api.nasdaq.com — same price data, no API key needed). '
                f'This is NOT yfinance.</span>'
            )
            + f'</div>'
            f'<hr style="border-color:#1E3A5F;margin:.4rem 0">'
            # Two-column: RapidAPI health check | TastyTrade
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:.3rem 1.2rem">'
            f'<div style="color:#FCD34D;font-weight:700;margin-bottom:.2rem">RAPIDAPI (TradingView) — HEALTH CHECK ONLY</div>'
            f'<div style="color:#FCD34D;font-weight:700;margin-bottom:.2rem">TASTYTRADE (OAuth) — OPTIONS BACKTEST</div>'
            f'<div style="color:#F1F5F9">'
            f'<span style="color:#93C5FD;font-size:.68rem">This section proves your RapidAPI key is active. '
            f'The market movers endpoint is pinged as a health check — it does NOT provide OHLCV bars '
            f'to the AI or backtest. Historical data chain is shown above.</span><br><br>'
            f'API Key Present: {_ab(_ro_key_ok)}<br>'
            f'Called This Run: {_ab(_ro_called)}<br>'
            f'HTTP Status: <b>{_ro_status if _ro_called else "---"}</b><br>'
            f'Endpoint: <code style="font-size:.66rem">{_ro_endpoint.replace("https://","")}</code><br>'
            f'Total Movers: <b>{_ro_count if _ro_count is not None else "---"}</b><br>'
            f'Top Symbols: <b>{", ".join(_ro_syms[:3]) if _ro_syms else "---"}</b> '
            f'<span style="color:#64748B;font-size:.66rem">(today\'s top movers — not related to your symbol)</span><br>'
            f'Role: <b>LIVE_MARKET_MOVERS / API_KEY_HEALTH_CHECK</b><br>'
            f'Used for Historical OHLCV: {_ab(False, "YES", "NO — see chain above")}<br>'
            f'Used for Gemini Historical Prompt: {_ab(_ro_leakage, "YES", "NO (leakage-safe)")}<br>'
            + (f'Error: <span style="color:#EF4444">{_ro_err}</span>' if _ro_err else "")
            + f'</div>'
            f'<div style="color:#F1F5F9">'
            f'Credential Source: <b>{_tt_cred_src}</b><br>'
            f'Refresh Token Present: {_ab(_tt_rt_present, "YES", "NO")}<br>'
            f'Access Token Present: {_ab(_tt_at_present, "YES", "NO")}<br>'
            f'Token Refresh Attempted: {_ab(_tt_ref_att, "YES", "NOT_RUN")}<br>'
            f'Token Refresh Status: {_stat_badge(_tt_ref_stat)}<br>'
            f'Customer Check Status: {_stat_badge(_tt_cust_stat)}<br>'
            f'Auth HTTP Status: <b>{"N/A" if _tt_http is None else _tt_http}</b><br>'
            f'Backtest Allowed: {_ab(_tt_allowed_ui)}<br>'
            f'<br><b style="color:#FCD34D">OPTIONS BACKTEST RESULT:</b><br>'
            f'Backtest Status: <b style="color:{"#10B981" if opts_status == "SUCCESS" else "#EF4444"}">'
            f'{opts_status}</b><br>'
            f'Backtest ID: <b>{opts.get("backtest_id") or "—"}</b><br>'
            + (
                f'Options P&L: <b style="color:{"#10B981" if float(opts.get("profit_loss",0) or 0)>=0 else "#EF4444"}">'
                f'${float(opts.get("profit_loss",0) or 0):+,.2f}</b><br>'
                f'Win Rate: <b>{f"{float(opts.get("win_rate",0) or 0)*100:.1f}%"}</b><br>'
                f'Trials / Trades: <b>{opts.get("total_trades","---")}</b><br>'
                f'Result Source: <b style="color:{"#FCD34D" if opts.get("bs_fallback") else "#10B981"}">'
                f'{"Black-Scholes (TT API failed)" if opts.get("bs_fallback") else "TastyTrade API (real data)"}</b><br>'
                if opts_status == "SUCCESS" else
                f'P&L: <b>N/A</b><br>Win Rate: <b>N/A</b><br>Trades: <b>N/A</b><br>'
            )
            + f'Accuracy Saved: {_ab(st.session_state.get("saved", False))}<br>'
            + (f'Auth/Backtest Reason: <span style="color:#FCA5A5;font-size:.67rem">{_tt_reason_ui}</span>' if not _tt_allowed_ui else "")
            + (f'<br>BS Fallback Reason: <span style="color:#FCD34D;font-size:.67rem">{opts.get("bs_fallback_reason","")}</span>' if opts.get("bs_fallback") and opts.get("bs_fallback_reason") else "")
            + (f'<br>TT API Error Body: <span style="color:#FCA5A5;font-size:.62rem;word-break:break-all">{str(opts.get("bs_fallback_body",""))[:600]}</span>' if opts.get("bs_fallback") and opts.get("bs_fallback_body") else "")
            + (f'<br>Backtest Error: <span style="color:#EF4444">{opts.get("error","")}</span>' if opts_status not in ("SUCCESS", "SKIPPED") else "")
            + f'</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── AI Prediction (left) | Options Backtest Actual (right) ───────────────
    st.markdown(
        f'<div style="color:#8BA9C4;font-size:.8rem;margin:.8rem 0 .2rem 0;font-weight:600">'
        f'<b>{symbol}</b> &nbsp;|&nbsp; Backtest: {origin} → {target} &nbsp;|&nbsp; '
        f'AI context: {ctx_start} → {origin} ({len([b for b in hist if ctx_start <= b["date"] <= origin])} bars)'
        f'</div>',
        unsafe_allow_html=True,
    )

    c3l, c3r = st.columns(2)

    with c3l:
        st.markdown(
            '<div style="background:#0F2940;padding:.5rem 1rem;border-radius:6px;'
            'color:#93C5FD;font-weight:900;font-size:.88rem;margin-bottom:.4rem;letter-spacing:.03em">'
            "AI PREDICTION</div>",
            unsafe_allow_html=True,
        )
        _ret_err_raw = cmp.get("return_error_pct") if cmp_ok else None
        _ret_err_fmt = (
            f'<span style="color:#EF4444;font-weight:700">{float(_ret_err_raw):+.2f}pp</span>'
            if _ret_err_raw is not None else "---"
        )
        _ai_dec = ai.get("decision", "---")
        _user_dir = opts_p.get("direction", "") if opts_p else ""
        st.markdown(
            _table([
                ("AI Market Signal",         _decision_badge(_ai_dec)),
                ("Origin Price Used",         _fmt(ai.get("origin_price_used"), "$")),
                ("Predicted Target Price",    _fmt(ai.get("predicted_target_price"), "$")),
                ("Predicted Return %",        _sign_fmt(ai.get("predicted_return_pct"), suffix="%")),
                ("Predicted Final Capital",   _fmt(ai.get("predicted_final_capital"), "$")),
                ("Predicted Total P&L",       _sign_fmt(ai.get("predicted_total_pl"))),
                ("Confidence Score",          f"{ai.get('confidence_score', '---')}/100"),
                ("Risk Score",               f"{ai.get('risk_score', '---')}/100"),
                ("Return Error vs Actual",    _ret_err_fmt),
                ("Leakage Check",             ai.get("leakage_check", "---")),
            ]),
            unsafe_allow_html=True,
        )
        # Explain AI signal vs user's chosen strategy — OPTIONS-AWARE alignment
        # "Sell Put" is a BULLISH/neutral position (profits when stock stays above put strike).
        # Comparing "SELL" direction directly to AI "BUY" would wrongly show CONFLICT.
        # Must map (direction, option_type) → market directional bias first.
        _conflict_note = ""
        if _user_dir and _ai_dec not in ("---", "HOLD"):
            _dir_lower = _user_dir.lower()
            _opt_type  = (opts_p.get("opt_type", "") or "").lower()
            _ai_upper  = _ai_dec.upper()

            # (direction, option_type) → market bias:
            # Sell Put  = neutral/bullish  → BUY bias
            # Buy Call  = bullish          → BUY bias
            # Sell Call = neutral/bearish  → SELL bias
            # Buy Put   = bearish          → SELL bias
            if _dir_lower == "sell" and _opt_type in ("put", "p"):
                _mkt_bias     = "BUY"
                _strategy_lbl = "Sell Put (neutral/bullish — profits when stock holds above put strike)"
            elif _dir_lower == "buy" and _opt_type in ("call", "c"):
                _mkt_bias     = "BUY"
                _strategy_lbl = "Buy Call (bullish — profits when stock rises above strike)"
            elif _dir_lower == "sell" and _opt_type in ("call", "c"):
                _mkt_bias     = "SELL"
                _strategy_lbl = "Sell Call (neutral/bearish — profits when stock stays below call strike)"
            elif _dir_lower == "buy" and _opt_type in ("put", "p"):
                _mkt_bias     = "SELL"
                _strategy_lbl = "Buy Put (bearish — profits when stock falls below put strike)"
            else:
                _mkt_bias     = _user_dir.upper()
                _strategy_lbl = f"{_user_dir.upper()} {_opt_type.upper()}" if _opt_type else _user_dir.upper()

            if (_mkt_bias == "BUY" and _ai_upper == "SELL") or (_mkt_bias == "SELL" and _ai_upper == "BUY"):
                _conflict_note = (
                    f'<span style="color:#F59E0B;font-weight:700">⚠ AI vs Strategy: CONFLICT</span> — '
                    f'Your strategy is <b>{_strategy_lbl}</b> (a <b>{_mkt_bias}</b>-biased position). '
                    f'AI independently predicts <b>{_ai_upper}</b> — the opposite market direction. '
                    f'AI does NOT read your strategy input; it analyzes market data separately. '
                    f'This CONFLICT means the market may move against your options position.'
                )
            elif (_mkt_bias == "BUY" and _ai_upper == "BUY") or (_mkt_bias == "SELL" and _ai_upper == "SELL"):
                _conflict_note = (
                    f'<span style="color:#34D399;font-weight:700">✓ AI Agrees with Strategy: ALIGNED</span> — '
                    f'Your strategy <b>{_strategy_lbl}</b> has a <b>{_mkt_bias}</b> bias that '
                    f'matches AI signal <b>{_ai_upper}</b>. '
                    f'AI analyzes the market independently — this alignment adds confidence to your position.'
                )
        if _conflict_note:
            _conflict_border = "#F59E0B" if "CONFLICT" in _conflict_note else "#34D399"
            st.markdown(
                f'<div style="background:#1E293B;border-left:3px solid {_conflict_border};'
                f'padding:.5rem .75rem;border-radius:4px;font-size:.72rem;'
                f'color:#CBD5E1;margin:.4rem 0">{_conflict_note}</div>',
                unsafe_allow_html=True,
            )

    with c3r:
        st.markdown(
            '<div style="background:#052E1A;padding:.5rem 1rem;border-radius:6px;'
            'color:#6EE7B7;font-weight:900;font-size:.88rem;margin-bottom:.4rem;letter-spacing:.03em">'
            "OPTIONS BACKTEST ACTUAL</div>",
            unsafe_allow_html=True,
        )
        if opts.get("exit_cond_dropped"):
            st.warning(
                "⚠️ Exit conditions (Stop Loss / Take Profit) were rejected by the TastyTrade API "
                "and auto-removed. Backtest ran WITHOUT TP/SL — trades held until expiry or end date. "
                "This causes results to differ significantly from the TastyTrade website. "
                "Try using only Take Profit OR only Stop Loss (not both) for better API compatibility."
            )
        _opts_dir_check = (st.session_state.get("opts_params") or {}).get("direction", "Sell")
        if _opts_dir_check.lower() == "buy" and not opts.get("exit_cond_dropped"):
            st.info(
                "ℹ️ BUY (Long) strategy detected: The TastyTrade backtester API may not apply "
                "Take Profit / Stop Loss exit conditions for BUY strategies in the same way "
                "as the TastyTrade website. If results differ from the website, this is the reason. "
                "The TastyTrade website uses actual historical option prices; the API may use a "
                "theoretical pricing model. For accurate BUY strategy results, verify on the website directly."
            )
        if opts_status == "SUCCESS":
            pnl        = opts.get("profit_loss")
            win_rate   = opts.get("win_rate")
            avg_pnl    = opts.get("avg_pnl")
            tot_trades = opts.get("total_trades")
            max_loss_v = opts.get("max_loss")
            max_prof_v = opts.get("max_profit")
            wr_fmt = (
                f"{float(win_rate)*100:.1f}%" if (win_rate is not None and float(win_rate) <= 1)
                else (f"{float(win_rate):.1f}%" if win_rate is not None else "---")
            )
            _agr       = cmp.get("agreement", "---") if cmp_ok else "---"
            _agr_color = "#EF4444" if _agr == "CONFLICT" else ("#10B981" if _agr == "AGREE" else "#9CA3AF")
            _agr_badge = f'<span style="color:{_agr_color};font-weight:800">{_agr}</span>'
            _bt_range_disp = opts.get("backtest_range", f"{ctx_start} → {target}")
            st.markdown(
                _table([
                    ("Options Backtest P&L",    _sign_fmt(pnl) if pnl is not None else "---"),
                    ("Win Rate",                wr_fmt),
                    ("Avg P&L / Trade",         f"${avg_pnl:+,.2f}" if avg_pnl is not None else "---"),
                    ("Total Trades",            str(tot_trades) if tot_trades is not None else "---"),
                    ("Max Single-Trade Loss",   _sign_fmt(max_loss_v) if max_loss_v is not None else "---"),
                    ("Max Single-Trade Profit", _sign_fmt(max_prof_v) if max_prof_v is not None else "---"),
                    ("Backtest Window",         _bt_range_disp),
                    ("AI vs Backtest",          _agr_badge if cmp_ok else "---"),
                    ("Backtest ID",             f'<code style="font-size:.65rem">{opts.get("backtest_id", "---")}</code>'),
                ]),
                unsafe_allow_html=True,
            )
            if max_loss_v is not None and max_loss_v < 0:
                _loss_abs = abs(max_loss_v)
                _cap_float = float(cap) if cap else 0
                if _cap_float > 0 and _loss_abs > _cap_float:
                    _err_card(
                        f"<b>RISK WARNING: Max single-trade loss (${_loss_abs:,.2f}) exceeds initial capital (${_cap_float:,.2f}).</b><br>"
                        "This indicates extreme leverage — options positions lost more than the capital allocated. "
                        "Review position sizing, contract quantity, and DTE before live trading."
                    )
                elif _loss_abs > 50_000:
                    _warn_card(
                        f"<b>RISK NOTE: Max single-trade loss was ${_loss_abs:,.2f}.</b><br>"
                        "Large individual trade losses detected. Review position sizing before live trading."
                    )
            # Trade count anomaly detection for infrequent schedules
            _sched = opts_p.get("entry_schedule", "Every day")
            if tot_trades is not None and _sched in ("Monthly", "Weekly"):
                _days_per_entry = 21 if _sched == "Monthly" else 5
                _expected_trades = max(1, int(horizon / _days_per_entry))
                _actual_trades = int(tot_trades or 0)
                if _actual_trades > _expected_trades * 3:
                    _warn_card(
                        f"<b>ENTRY FREQUENCY ANOMALY: {_sched} schedule returned {_actual_trades} trades "
                        f"in {horizon} days (expected ~{_expected_trades}).</b><br>"
                        f"Tastytrade may not recognize the '{_sched.lower()}' frequency string — "
                        "the API may be defaulting to daily entry. "
                        "P&L results reflect more entries than intended. "
                        "Verify with <code>verify_tastytrade_delta_backtest.py --entry-frequency monthly</code>."
                    )
        elif opts_status == "SKIPPED":
            _skip_err  = opts.get("error", "")
            _skip_note = opts.get("note", "")
            _skip_body = _skip_err or _skip_note or "Options backtest was not run."
            _cred_hint = (
                "<br>Check TASTYTRADE_USERNAME / TASTYTRADE_PASSWORD in .env."
                if _skip_err and "credential" in _skip_err.lower()
                else ""
            )
            _warn_card(
                f"<b>Options backtest skipped.</b><br>"
                f"{_skip_body}{_cred_hint}"
            )
        elif opts_status == "FLAT_NO_TRADES":
            _dte_hint = opts.get("effective_dte_used") or opts.get("payload_dte") or (opts_p.get("dte", "?") if opts_p else "?")
            _bt_range_s = opts.get("backtest_range", f"{origin} → {target}")
            try:
                import datetime as _fnt_dt
                _fnt_parts = _bt_range_s.replace("→", " ").split()
                _fnt_start, _fnt_end = _fnt_parts[0], _fnt_parts[-1]
                _fnt_window_days = (_fnt_dt.date.fromisoformat(_fnt_end) - _fnt_dt.date.fromisoformat(_fnt_start)).days
            except Exception:
                _fnt_window_days = None
            _dte_int = int(_dte_hint) if str(_dte_hint).isdigit() else None
            _short_window = _fnt_window_days is not None and _fnt_window_days < 7
            _short_dte    = _dte_int is not None and _dte_int < 7
            if _short_window and _short_dte:
                _fnt_cause = (
                    f"The backtest window is only <b>{_fnt_window_days} day(s)</b> "
                    f"AND DTE={_dte_hint} day(s) is too small. Both need to be fixed."
                )
                _fnt_fix = (
                    "Set <b>Decision Horizon to at least 14 days</b> and "
                    "<b>DTE to at least 14 days</b> (recommended 14–45)."
                )
            elif _short_window:
                _fnt_cause = (
                    f"The backtest window is only <b>{_fnt_window_days} day(s)</b> ({_bt_range_s}). "
                    f"A DTE={_dte_hint} position needs time to open AND reach TP/SL or expiry. "
                    "TastyTrade cannot complete any trades in a 1–6 day window."
                )
                _fnt_fix = (
                    "Set <b>Decision Horizon to at least 14 days</b> so the backtest window is wide enough "
                    "for option positions to complete."
                )
            elif _short_dte:
                _fnt_cause = (
                    f"DTE={_dte_hint} day(s) is too small. "
                    "TastyTrade's historical database rarely contains option contracts "
                    "with fewer than 7 days to expiration."
                )
                _fnt_fix = "Set <b>Expiration (DTE) to at least 7 days</b> (recommended: 14–45)."
            else:
                _fnt_cause = (
                    f"TastyTrade found no matching option contracts for DTE={_dte_hint}, "
                    f"delta=30 within the backtest window {_bt_range_s}. "
                    "This can happen when options data is unavailable for the specific date or symbol."
                )
                _fnt_fix = (
                    "Try a <b>longer backtest window</b> (more days between origin and target), "
                    "a <b>larger DTE</b> (14–45), or verify TSLA options exist on those dates."
                )
            _err_card(
                f"<b>Tastytrade authentication succeeded. Backtest returned zero trades.</b><br>"
                "This is a <b>parameter / contract availability</b> issue — not a credential issue.<br><br>"
                f"<b>Root cause:</b> {_fnt_cause}<br><br>"
                f"<b>Fix:</b> {_fnt_fix}"
            )
        elif opts_status == "RATE_LIMITED":
            _retry_s = opts.get("retry_after", "unknown")
            _err_card(
                f"<b>Tastytrade rate limit reached (HTTP 429).</b><br>"
                f"Too many requests sent to Tastytrade backtester. "
                f"<b>Retry-After: {_retry_s}</b> — wait before retrying.<br>"
                "Accuracy NOT saved. No stale result shown. Backtest did NOT run."
            )
        elif opts_status == "EXACT_STRIKE_UNSUPPORTED":
            _warn_card(
                f"<b>Exact strike not supported by provider — backtest NOT run.</b><br>"
                f"Tastytrade does not support fixed-strike backtesting. Requested strike: "
                f"<b>{opts.get('requested_strike', '—')}</b>.<br>"
                "To run an approximate reference using delta proxy (ATM ~50), check "
                "<b>'Allow delta proxy approximation'</b> in the form above.<br>"
                "Accuracy NOT saved. No payload was sent."
            )
        elif opts_status == "BACKTEST_CREATE_FAILED":
            _http_code = opts.get("http_status", "")
            _err_body  = opts.get("error_body", "")
            _err_card(
                f"<b>Options backtest failed: HTTP {_http_code}.</b><br>"
                "Tastytrade authentication succeeded — this is a <b>payload/provider validation issue</b>, "
                "not a credential issue.<br>"
                f"{opts.get('error', '')}.<br>"
                + (f'<br><span style="color:#9CA3AF;font-size:.72rem">Response body: {_err_body[:400]}</span>' if _err_body else "")
            )
        elif opts_status == "PENDING":
            _bt_range = opts.get("backtest_range", "")
            _reason   = opts.get("reason", "Target date is in the future.")
            st.info(
                f"**Options backtest PENDING — target date is in the future.**\n\n"
                f"{_reason}\n\n"
                f"**Backtest window:** {_bt_range}\n\n"
                f"The TastyTrade options backtest runs on **historical data only**. "
                f"Your prediction window extends into the future, so no options data exists yet to backtest against. "
                f"This prediction has been saved and will be validated automatically once the target date passes.\n\n"
                f"**To get immediate backtest results:** Use a Prediction Origin Date that is **at least 30+ days in the past** "
                f"(e.g. if today is 2026-07-25 and horizon is 30 days, use origin ≤ 2026-06-25)."
            )
        else:
            _err_card(
                f"<b>Options backtest {opts_status}.</b><br>"
                f"{opts.get('error', 'Unknown error')}."
            )

    # AI reasoning — full structured breakdown
    _ai_main2    = ai.get("reasoning") or ai.get("main_reason") or ""
    _ai_trend2   = ai.get("trend_assessment", "") or ""
    _ai_mom2     = ai.get("momentum_assessment", "") or ""
    _ai_vol2     = ai.get("volatility_assessment", "") or ""
    _ai_bench2   = ai.get("benchmark_assessment", "") or ""
    _ai_bull2    = ai.get("bull_case", "") or ""
    _ai_bear2    = ai.get("bear_case", "") or ""
    _ai_why_not2 = ai.get("why_not_opposite", "") or ""
    _ai_inv2     = ai.get("invalidating_conditions", []) or []
    _ai_kf2      = ai.get("key_features_used", []) or []
    _ai_opts2    = ai.get("options_strategy_assessment", "") or ""
    if _ai_main2 or _ai_trend2:
        _reasoning_rows2 = ""
        if _ai_opts2:
            _reasoning_rows2 += (
                f'<div style="margin-bottom:.7rem;padding:.5rem .8rem;'
                f'background:#1C1009;border-left:3px solid #F59E0B;border-radius:4px">'
                f'<span style="color:#FCD34D;font-weight:700">Options Strategy Assessment:</span> {_ai_opts2}</div>'
            )
        if _ai_main2:
            _reasoning_rows2 += f'<div style="margin-bottom:.6rem"><span style="color:#93C5FD;font-weight:700">Prediction Rationale:</span> {_ai_main2}</div>'
        if _ai_trend2:
            _reasoning_rows2 += f'<div style="margin-bottom:.4rem"><span style="color:#6EE7B7;font-weight:600">Trend:</span> {_ai_trend2}</div>'
        if _ai_mom2:
            _reasoning_rows2 += f'<div style="margin-bottom:.4rem"><span style="color:#6EE7B7;font-weight:600">Momentum (RSI/MACD):</span> {_ai_mom2}</div>'
        if _ai_vol2:
            _reasoning_rows2 += f'<div style="margin-bottom:.4rem"><span style="color:#6EE7B7;font-weight:600">Volatility:</span> {_ai_vol2}</div>'
        if _ai_bench2:
            _reasoning_rows2 += f'<div style="margin-bottom:.4rem"><span style="color:#6EE7B7;font-weight:600">vs Benchmark:</span> {_ai_bench2}</div>'
        if _ai_bull2:
            _reasoning_rows2 += f'<div style="margin-bottom:.4rem"><span style="color:#10B981;font-weight:600">Bull Case:</span> {_ai_bull2}</div>'
        if _ai_bear2:
            _reasoning_rows2 += f'<div style="margin-bottom:.4rem"><span style="color:#EF4444;font-weight:600">Bear Case:</span> {_ai_bear2}</div>'
        if _ai_why_not2:
            _reasoning_rows2 += f'<div style="margin-bottom:.4rem"><span style="color:#F59E0B;font-weight:600">Why Not Opposite:</span> {_ai_why_not2}</div>'
        if _ai_kf2:
            _reasoning_rows2 += f'<div style="margin-bottom:.4rem"><span style="color:#C4B5FD;font-weight:600">Key Signals Used:</span> {", ".join(_ai_kf2)}</div>'
        if _ai_inv2:
            _reasoning_rows2 += f'<div style="margin-top:.4rem"><span style="color:#F87171;font-weight:600">Invalidating Conditions:</span> {" | ".join(_ai_inv2)}</div>'
        st.markdown(
            f'<div style="background:#0A1929;border-left:4px solid #3B82F6;border-radius:6px;'
            f'padding:.9rem 1.2rem;margin:.6rem 0;font-size:.82rem;color:#CBD5E1;line-height:1.6">'
            f'{_reasoning_rows2}</div>',
            unsafe_allow_html=True,
        )

    # ── Strategy Parameters (collapsed) ──────────────────────────────────────
    _3b_bt_range = opts.get("backtest_range", f"{ctx_start} → {target}")
    with st.expander(f"Strategy Parameters  |  Backtest: {_3b_bt_range}", expanded=False):
        oc1, oc2, oc3 = st.columns(3)
        with oc1:
            st.markdown(
                _table([
                    ("Symbol",     symbol),
                    ("Direction",  opts_p.get("direction", "---")),
                    ("Type",       opts_p.get("opt_type", "---")),
                    ("Quantity",   f"{opts_p.get('quantity', '---')} contract(s)"),
                ]),
                unsafe_allow_html=True,
            )
        with oc2:
            _3b_sel = opts_p.get("strike_selection", "delta")
            _3b_is_strike = _3b_sel == "strike"
            _3b_contract_row = (
                ("Strike",  str(opts_p.get("strike_price", "---")))
                if _3b_is_strike else
                ("Delta",   str(opts_p.get("delta", "---")))
            )
            _3b_extra = (
                [("Expiry Date", opts_p.get("expiry_date") or "---")]
                if _3b_is_strike else []
            )
            st.markdown(
                _table([
                    ("Strike Selection", _3b_sel.title()),
                    _3b_contract_row,
                    ("DTE",              f"{opts_p.get('dte', '---')} days"),
                ] + _3b_extra),
                unsafe_allow_html=True,
            )
        with oc3:
            st.markdown(
                _table([
                    ("Entry Schedule",   opts_p.get("entry_schedule", "---")),
                    ("Take Profit",     f"{opts_p.get('take_profit_pct')}% of premium" if opts_p.get("take_profit_pct") else "off"),
                    ("Stop Loss",       f"{opts_p.get('stop_loss_pct')}% of premium"   if opts_p.get("stop_loss_pct")   else "off"),
                    ("Exit After Days", str(opts_p.get("exit_after_days")) + " days"   if opts_p.get("exit_after_days") else "off"),
                    ("Backtest Start",  origin),
                    ("Backtest End",    target),
                ]),
                unsafe_allow_html=True,
            )

    # ── Strike distance and DTE/expiry warnings ────────────────────────────────
    _sd_status = opts.get("strike_distance_status", "")
    _sd_warn   = opts.get("strike_distance_warning", "")
    _dte_status = opts.get("dte_expiry_status", "")
    _dte_warn   = opts.get("dte_expiry_mismatch_warning", "")
    if _sd_status in ("EXTREME_STRIKE_DISTANCE", "FAR_STRIKE"):
        _err_card(f"<b>Strike Distance Warning:</b> {_sd_warn}")
    elif _sd_status == "FAR_OTM_OR_ITM_STRIKE":
        _warn_card(f"<b>Strike Distance:</b> {_sd_warn or 'Strike is far OTM/ITM — contract may be illiquid.'}")
    if _dte_status == "DTE_EXPIRY_MISMATCH":
        _warn_card(
            f"<b>DTE/Expiry Mismatch:</b> {_dte_warn}<br>"
            f"Effective DTE used: <b>{opts.get('effective_dte_used')}</b> "
            f"(computed from expiry date). User-entered DTE was: {opts_p.get('dte')}."
        )
    if opts.get("exact_validation_status") == "UNSUPPORTED_BY_PROVIDER":
        _warn_card(
            f"<b>Exact Strike Unsupported:</b> {opts.get('exact_validation_note', '')}.<br>"
            "Accuracy is NOT saved for exact-strike runs. Delta proxy shown as reference only."
        )

    # ── Options Backtest Numeric Summary ─────────────────────────────────────
    _section_header(
        None, "Options Backtest Summary",
        "Tastytrade backtest result — AI options prediction vs actual options P&L",
    )

    if opts_status == "SUCCESS":
        pnl_v      = float(opts.get("profit_loss") or 0)
        win_rate_v = float(opts.get("win_rate") or 0)
        avg_pnl_v  = float(opts.get("avg_pnl") or 0)
        n_trades_v = int(opts.get("total_trades") or 0)
        wr_str     = f"{win_rate_v*100:.1f}%" if win_rate_v <= 1 else f"{win_rate_v:.1f}%"

        mc = st.columns(4)
        mc[0].metric("Options Total P&L",  f"${pnl_v:+,.2f}")
        mc[1].metric("Win Rate",           wr_str)
        mc[2].metric("Avg P&L / Trade",    f"${avg_pnl_v:+,.2f}")
        mc[3].metric("Total Trades",       str(n_trades_v))

    elif opts_status == "FLAT_NO_TRADES":
        _dte_used  = opts.get("effective_dte_used") or opts.get("payload_dte") or "?"
        _bt_range  = opts.get("backtest_range", f"{origin} → {target}")
        try:
            import datetime as _sum_dt
            _sum_parts = _bt_range.replace("→", " ").split()
            _sum_start, _sum_end = _sum_parts[0], _sum_parts[-1]
            _sum_window_days = (_sum_dt.date.fromisoformat(_sum_end) - _sum_dt.date.fromisoformat(_sum_start)).days
        except Exception:
            _sum_window_days = None
        _sum_dte_int   = int(_dte_used) if str(_dte_used).isdigit() else None
        _sum_short_win = _sum_window_days is not None and _sum_window_days < 7
        _sum_short_dte = _sum_dte_int is not None and _sum_dte_int < 7
        if _sum_short_win and _sum_short_dte:
            _sum_why = (
                f"• <b>Backtest window = {_sum_window_days} day(s)</b> — too short for any position to open and close.<br>"
                f"• <b>DTE = {_dte_used} day(s)</b> — TastyTrade rarely has contracts this short-dated in its historical data."
            )
            _sum_fix = f"Set <b>Decision Horizon ≥ 14 days</b> and <b>DTE ≥ 14 days</b> (recommended 14–45)."
        elif _sum_short_win:
            _sum_why = (
                f"• <b>Backtest window = {_sum_window_days} day(s)</b> ({_bt_range}) — too short. "
                f"A DTE={_dte_used} position needs enough time to open AND reach TP/SL or expiry. "
                "TastyTrade cannot complete any trades in fewer than 7 days."
            )
            _sum_fix = "Set <b>Decision Horizon to at least 14 days</b> so option positions have time to complete."
        elif _sum_short_dte:
            _sum_why = (
                f"• <b>DTE = {_dte_used} day(s)</b> — TastyTrade's historical database rarely contains "
                "option contracts with fewer than 7 days to expiration."
            )
            _sum_fix = "Set <b>Expiration (DTE) to at least 7 days</b> (recommended: 14–45)."
        else:
            _sum_why = (
                f"• No matching option contracts found for DTE={_dte_used}, delta=30 within "
                f"the backtest window {_bt_range}. Options data may not be available for this symbol on those dates."
            )
            _sum_fix = "Try a longer backtest window, a different DTE (14–45), or verify options existed on these dates."
        _err_card(
            f"<b>Options Backtest FLAT_NO_TRADES — TastyTrade ran the backtest but found ZERO matching contracts.</b><br>"
            f"Backtest window: <b>{_bt_range}</b> &nbsp;|&nbsp; DTE used: <b>{_dte_used}</b><br><br>"
            f"<b>Why this happens:</b><br>"
            f"{_sum_why}<br><br>"
            f"<b>Fix:</b> {_sum_fix}<br>"
            "Comparison and accuracy save are BLOCKED until backtest returns trades."
        )
    elif opts_status == "PENDING":
        _bt_range = opts.get("backtest_range", "")
        st.info(
            f"**Options Backtest PENDING — target date is in the future.**\n\n"
            f"Backtest window: **{_bt_range}** is in the future. "
            f"The TastyTrade backtester only works on **historical data**. "
            f"This prediction is saved as PENDING and will be validated on/after the target date.\n\n"
            f"**Fix:** Use a Prediction Origin Date at least 30+ days in the past so the entire "
            f"prediction window (origin → target) is historical."
        )
    else:
        _err_card(
            f"<b>Options Backtest {opts_status} — cannot compare AI vs backtest.</b><br>"
            f"{opts.get('error', 'Unknown error')}.<br>"
            "<b>Comparison and accuracy save are BLOCKED until backtest succeeds.</b>"
        )

    # ── TastyTrade-Style Full Results (Summary / Details / Logs) ─────────────
    if opts_status == "SUCCESS":
        _section_header(None, "Backtest Results", "Matching TastyTrade output — Summary, Details, Logs")

        _raw_trials   = opts.get("trials", [])
        _raw_stats    = opts.get("raw_stats", {})
        _raw_bt_data  = opts.get("raw_bt_data", {})
        _res_raw      = (_raw_bt_data.get("results") or {}) if _raw_bt_data else {}

        # Daily settlement from API (may be nested under results)
        _daily_rows = (
            _res_raw.get("dailySettlements")
            or _res_raw.get("daily_settlements")
            or _res_raw.get("dailysettlements")
            or []
        )
        # Transactions from API
        _txn_rows = (
            _res_raw.get("transactions")
            or _res_raw.get("orders")
            or []
        )

        # ── Strategy Config Recap (TastyTrade header style) ──────────────────
        _tp_disp  = opts_p.get("take_profit_pct")
        _sl_disp  = opts_p.get("stop_loss_pct")
        _ead_disp = opts_p.get("exit_after_days")
        _exit_parts = []
        if _tp_disp: _exit_parts.append(f"Take profit: {_tp_disp}% of premium")
        if _sl_disp: _exit_parts.append(f"Stop loss: {_sl_disp}% of premium")
        if _ead_disp: _exit_parts.append(f"Exit after {_ead_disp} days")
        if not _exit_parts: _exit_parts.append(opts_p.get("exit_rule", "Exit at target date"))
        _exit_str = " | ".join(_exit_parts)
        _delta_disp = opts_p.get("delta", "30")
        _dte_disp   = opts_p.get("dte", "1")
        _dir_disp   = opts_p.get("direction", "Buy").lower()
        _type_disp  = opts_p.get("opt_type", "Call").lower()
        _qty_disp   = opts_p.get("quantity", 1)

        st.markdown(
            f'<div style="background:#0D1B2A;border:1px solid #1E4D7A;border-radius:8px;'
            f'padding:.7rem 1.2rem;margin:.4rem 0 .6rem 0;font-size:.8rem">'
            f'<span style="color:#FCD34D;font-weight:800">UNDERLYING:</span> '
            f'<span style="color:#F1F5F9;font-weight:700">{symbol}</span> &nbsp;|&nbsp; '
            f'<span style="color:#FCD34D;font-weight:800">LEGS:</span> '
            f'<span style="color:#6EE7B7">{_dir_disp} {_qty_disp} {_type_disp} {_delta_disp}Δ {_dte_disp} DTE</span> '
            f'&nbsp;|&nbsp; '
            f'<span style="color:#FCD34D;font-weight:800">ENTRY:</span> '
            f'<span style="color:#F1F5F9">{opts_p.get("entry_schedule","Every day")}</span> '
            f'&nbsp;|&nbsp; '
            f'<span style="color:#FCD34D;font-weight:800">EXIT:</span> '
            f'<span style="color:#F1F5F9">{_exit_str}</span> '
            f'&nbsp;|&nbsp; '
            f'<span style="color:#FCD34D;font-weight:800">DATES:</span> '
            f'<span style="color:#F1F5F9">{origin} → {target}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        _tt_tab_sum, _tt_tab_agent, _tt_tab_det, _tt_tab_log = st.tabs(["Summary", "🎯 Agent vs Backtester", "Details", "Logs"])

        # ── SUMMARY TAB ───────────────────────────────────────────────────────
        with _tt_tab_sum:
            import plotly.graph_objects as go

            # Build dual-axis chart: stock price (left) + strategy P&L (right)
            _chart_dates  = []
            _chart_prices = []
            _chart_pl     = []

            # Stock price curve from hist — show full backtest window (ctx_start to target)
            _hist_in_window = [b for b in hist if ctx_start <= b["date"] <= target]
            for b in sorted(_hist_in_window, key=lambda x: x["date"]):
                _chart_dates.append(b["date"])
                _chart_prices.append(float(b.get("close") or b.get("Close") or 0))

            # Strategy P&L curve — from daily settlements if available, else from cumulative trials
            if _daily_rows:
                _ds_dates = []
                _ds_pl    = []
                for dr in _daily_rows:
                    _ds_d = dr.get("date") or dr.get("Date") or ""
                    _ds_p = float(dr.get("totalProfitLoss") or dr.get("total_profit_loss")
                                  or dr.get("Total profit/loss") or 0)
                    if _ds_d:
                        _ds_dates.append(_ds_d)
                        _ds_pl.append(_ds_p)
            else:
                # Build cumulative from trials
                _trial_sorted = sorted(_raw_trials, key=lambda t: str(t.get("openDateTime") or t.get("entryDate") or t.get("entry_date") or ""))
                _cum = 0.0
                _ds_dates = []
                _ds_pl    = []
                for _t in _trial_sorted:
                    _exit_d = str(_t.get("closeDateTime") or _t.get("exitDate") or _t.get("exit_date") or "")[:10]
                    _pl_t   = float(_t.get("profitLoss") or _t.get("profit_loss") or 0)
                    _cum   += _pl_t
                    if _exit_d:
                        _ds_dates.append(_exit_d)
                        _ds_pl.append(_cum)

            fig = go.Figure()

            if _chart_dates and _chart_prices:
                fig.add_trace(go.Scatter(
                    x=_chart_dates, y=_chart_prices,
                    name=f"{symbol} end of day price",
                    line=dict(color="#94A3B8", width=2),
                    yaxis="y1",
                ))

            if _ds_dates and _ds_pl:
                fig.add_trace(go.Scatter(
                    x=_ds_dates, y=_ds_pl,
                    name="Strategy profit/loss",
                    line=dict(color="#F97316", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(249,115,22,0.08)",
                    yaxis="y2",
                ))

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0D1B2A",
                plot_bgcolor="#0D1B2A",
                margin=dict(l=60, r=60, t=30, b=40),
                height=320,
                legend=dict(orientation="h", x=0, y=1.12, font=dict(size=11)),
                yaxis=dict(
                    title=dict(text=f"{symbol} Price ($)", font=dict(color="#94A3B8")),
                    tickfont=dict(color="#94A3B8"),
                    side="left",
                ),
                yaxis2=dict(
                    title=dict(text="Strategy P&L ($)", font=dict(color="#F97316")),
                    tickfont=dict(color="#F97316"),
                    overlaying="y",
                    side="right",
                    tickprefix="$",
                ),
                xaxis=dict(tickfont=dict(color="#94A3B8")),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Show data source banner
            _is_bs_result  = opts.get("bs_fallback") or opts.get("bs_source") in ("black_scholes_simulation", "black_scholes_sl_correction")
            _bs_sl_fixed   = opts.get("bs_sl_corrected", False)
            _bs_sl_val_disp = opts.get("bs_sl_value", 0)
            _bs_sigma_disp = opts.get("bs_sigma_pct") or 0
            _sym_upper = str(symbol).upper() if symbol else ""
            if _is_bs_result:
                if _sym_upper in ("SPX", "XSP"):
                    st.error(
                        f"🚨 **SPX results WILL NOT match TastyTrade website** — "
                        f"The TastyTrade backtester API does not return real options prices for {_sym_upper} index options "
                        f"(it returns zero prices). Our platform falls back to Black-Scholes theoretical pricing, which is a "
                        f"mathematical model — not real market prices. This means trade P&Ls, premiums, and total results "
                        f"**will differ significantly from what you see on TastyTrade website.**\n\n"
                        f"✅ **FIX: Change your symbol from {_sym_upper} to SPY** — the TastyTrade API returns real "
                        f"options chain prices for SPY (ETF options), and results will match the website closely. "
                        f"SPY = S&P 500 ETF, tracks same index as SPX at ~1/10 the price level."
                    )
                elif _bs_sl_fixed:
                    st.info(
                        f"✅ **Stop Loss Auto-Corrected — Results are accurate** \n\n"
                        f"The TastyTrade API ignored your {_bs_sl_val_disp:.0f}% Stop Loss for this Sell strategy "
                        f"(a known API limitation). This platform automatically recomputed the backtest using "
                        f"Black-Scholes simulation with your exact {_bs_sl_val_disp:.0f}% SL correctly applied. "
                        f"The results below reflect trades being properly stopped out at the {_bs_sl_val_disp:.0f}% threshold — "
                        f"these numbers are realistic and can be used for analysis."
                    )
                else:
                    st.info(
                        f"**Black-Scholes Computed Results** — "
                        f"The TastyTrade API returned zero prices for this configuration. "
                        f"Results below are computed using Black-Scholes theoretical pricing"
                        + (f" with **{_bs_sigma_disp:.1f}% realized volatility**" if _bs_sigma_disp else "")
                        + f" from the underlying price history. "
                        f"Option prices and P&L are theoretical approximations — actual market prices may differ."
                    )
            else:
                _tt_sl_warn = opts.get("tt_api_sl_not_applied", False)
                _tt_sl_val  = opts.get("tt_api_sl_value", 0)
                if _tt_sl_warn:
                    # Fallback: BS correction failed — still flag it
                    _delta_disp = opts_p.get("delta_ui") or opts_p.get("delta") or "—"
                    _dte_disp   = opts_p.get("dte") or "—"
                    _tp_disp    = opts_p.get("take_profit_pct") or "—"
                    st.warning(
                        f"⚠️ **Stop Loss correction could not be applied for this run** — "
                        f"The TastyTrade API ignored your {_tt_sl_val:.0f}% SL and Black-Scholes correction "
                        f"encountered an error. The results shown may be inflated. "
                        f"Try re-running the backtest — the auto-correction will retry."
                    )

            # Your Strategy stats vs Buy and Hold
            _pnl_v    = float(opts.get("profit_loss") or 0)
            _raw_max_dd = float(_raw_stats.get("Max drawdown") or _raw_stats.get("maxDrawdown") or 0)
            # Always use user's stated initial capital as the capital baseline — TastyTrade website
            # uses the user's account balance (not API-computed BPR) for Used capital / ROC / CAGR.
            _used_cap = cap if cap else float(_raw_stats.get("Used capital") or _raw_stats.get("usedCapital") or 1)
            _roc      = (_pnl_v / _used_cap * 100) if _used_cap else 0
            _mar      = float(_raw_stats.get("MAR ratio") or _raw_stats.get("marRatio") or 0)

            # Buy and Hold: compute over the BACKTEST window only (origin → target), matching TastyTrade website.
            # ctx_start is the AI learning window start (much earlier); using it inflates Buy & Hold return.
            _bh_start = next((float(b.get("close") or 0) for b in sorted(hist, key=lambda x: x["date"]) if b["date"] >= origin), 0)
            _bh_end   = next((float(b.get("close") or 0) for b in sorted(hist, key=lambda x: x["date"], reverse=True) if b["date"] <= target), 0)
            _bh_pl    = ((_bh_end / _bh_start - 1) * _used_cap) if (_bh_start > 0 and _used_cap > 0) else 0
            _bh_ret   = ((_bh_end / _bh_start - 1) * 100) if _bh_start > 0 else 0

            cs1, cs2 = st.columns(2)
            with cs1:
                pnl_color = "#10B981" if _pnl_v >= 0 else "#EF4444"
                st.markdown(
                    f'<div style="background:#0A2540;border:1px solid #1E4D7A;border-radius:8px;padding:1rem 1.2rem">'
                    f'<div style="color:#93C5FD;font-weight:800;font-size:.85rem;margin-bottom:.7rem">Your Strategy</div>'
                    f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:.4rem">'
                    f'<div style="color:#94A3B8;font-size:.78rem">Total profit/loss</div>'
                    f'<div style="color:{pnl_color};font-weight:800;font-size:.9rem">${_pnl_v:+,.2f}</div>'
                    f'<div style="color:#94A3B8;font-size:.78rem">Max drawdown</div>'
                    f'<div style="color:#EF4444;font-weight:700">{_raw_max_dd:.2f}%</div>'
                    f'<div style="color:#94A3B8;font-size:.78rem">Return on used capital</div>'
                    f'<div style="color:{"#10B981" if _roc >= 0 else "#EF4444"};font-weight:700">{_roc:.1f}%</div>'
                    f'<div style="color:#94A3B8;font-size:.78rem">MAR ratio</div>'
                    f'<div style="color:#F1F5F9;font-weight:700">{_mar:.2f}</div>'
                    f'<div style="color:#94A3B8;font-size:.78rem">Used capital</div>'
                    f'<div style="color:#F1F5F9">${_used_cap:,.2f}</div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
            with cs2:
                bh_color = "#10B981" if _bh_pl >= 0 else "#EF4444"
                st.markdown(
                    f'<div style="background:#0A2540;border:1px solid #374151;border-radius:8px;padding:1rem 1.2rem">'
                    f'<div style="color:#9CA3AF;font-weight:800;font-size:.85rem;margin-bottom:.7rem">Buy and Hold ({symbol})</div>'
                    f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:.4rem">'
                    f'<div style="color:#94A3B8;font-size:.78rem">Total profit/loss</div>'
                    f'<div style="color:{bh_color};font-weight:800;font-size:.9rem">${_bh_pl:+,.2f}</div>'
                    f'<div style="color:#94A3B8;font-size:.78rem">Return on capital</div>'
                    f'<div style="color:{"#10B981" if _bh_ret >= 0 else "#EF4444"};font-weight:700">{_bh_ret:.2f}%</div>'
                    f'<div style="color:#94A3B8;font-size:.78rem">Start price</div>'
                    f'<div style="color:#F1F5F9">${_bh_start:,.2f}</div>'
                    f'<div style="color:#94A3B8;font-size:.78rem">End price</div>'
                    f'<div style="color:#F1F5F9">${_bh_end:,.2f}</div>'
                    f'<div style="color:#94A3B8;font-size:.78rem">Capital used</div>'
                    f'<div style="color:#F1F5F9">${_used_cap:,.2f}</div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                '<div style="color:#64748B;font-size:.72rem;text-align:center;margin-top:.5rem">'
                'Trades (entries and exits) are executed at the mid price, once per day, 15 minutes before market close.'
                '</div>',
                unsafe_allow_html=True,
            )

        # ── DETAILS TAB ──────────────────────────────────────────────────────
        with _tt_tab_det:
            _n_tr   = int(opts.get("total_trades") or 0)
            _n_wins = int(
                _raw_stats.get("Wins") or _raw_stats.get("numWins") or _raw_stats.get("num_wins")
                or _raw_stats.get("Trades with profits") or 0
            )
            _n_loss = int(
                _raw_stats.get("Losses") or _raw_stats.get("numLosses") or _raw_stats.get("num_losses")
                or _raw_stats.get("Trades with losses") or 0
            )
            # Fallback: if still 0 but we have trades, compute from trials
            if _n_wins == 0 and _n_loss == 0 and _n_tr > 0 and _raw_trials:
                _n_wins = sum(1 for _t in _raw_trials if float(_t.get("profitLoss") or _t.get("profit_loss") or 0) > 0)
                _n_loss = _n_tr - _n_wins
            _pr_rate = (_n_wins / _n_tr * 100) if _n_tr > 0 else 0
            _lr_rate = (_n_loss / _n_tr * 100) if _n_tr > 0 else 0
            _lg_prof = float(_raw_stats.get("Highest profit") or _raw_stats.get("maxProfit") or 0)
            _lg_loss = float(_raw_stats.get("Worst loss") or _raw_stats.get("maxLoss") or 0)
            # "Avg. return per trade" from API is a percentage (e.g. -9.35); display as %.
            # "Avg. profit/loss per trade" is a dollar amount; compute from total_pl/n_trades if missing.
            _avg_ret_pct = float(_raw_stats.get("Avg. return per trade") or _raw_stats.get("avgReturnPerTrade") or 0)
            _avg_pnl_dollar = float(
                _raw_stats.get("Avg. profit/loss per trade") or _raw_stats.get("avgProfitLoss") or opts.get("avg_pnl") or 0
            )
            if not _avg_pnl_dollar and _n_tr > 0:
                _avg_pnl_dollar = float(opts.get("profit_loss") or 0) / _n_tr
            _avg_days = float(_raw_stats.get("Avg. days in trade") or _raw_stats.get("avgDaysInTrade") or 0)
            _avg_bpr  = float(_raw_stats.get("Avg. BPR per trade") or _raw_stats.get("avgBuyingPowerReduction") or 0)
            _avg_prem = float(_raw_stats.get("Avg. premium") or _raw_stats.get("avgPremium") or 0)
            _avg_win  = float(_raw_stats.get("Avg. win size") or _raw_stats.get("avgWinSize") or 0)
            _avg_ls   = float(_raw_stats.get("Avg. loss size") or _raw_stats.get("avgLossSize") or 0)
            _tot_prem = float(_raw_stats.get("Total premium") or _raw_stats.get("totalPremium") or 0)
            _tot_fees = float(_raw_stats.get("Total fees") or _raw_stats.get("totalFees") or 0)
            _cagr_v   = float(_raw_stats.get("CAGR") or _raw_stats.get("cagr") or 0)
            _used_c   = cap if cap else float(_raw_stats.get("Used capital") or _raw_stats.get("usedCapital") or 1)

            _opts_qty = int((st.session_state.get("opts_params") or {}).get("quantity", 1))
            _opts_direction = (st.session_state.get("opts_params") or {}).get("direction", "Sell")
            # For BUY (long) strategies: API returns per-1-contract premium. Multiply by qty for total position.
            # For SELL (short) strategies: API already returns total-position premium. Do NOT multiply again.
            if _opts_qty > 1 and _avg_prem != 0 and _opts_direction.lower() in ("buy", "long"):
                _avg_prem *= _opts_qty
            if _opts_qty > 1 and _tot_prem != 0 and _opts_direction.lower() in ("buy", "long"):
                _tot_prem *= _opts_qty

            # For SELL strategies the API returns Total premium as the credit received (should be positive).
            # If the API returns a negative value for a SELL strategy, flip the sign.
            if _opts_direction.lower() == "sell" and _tot_prem < 0:
                _tot_prem = abs(_tot_prem)
                _avg_prem = abs(_avg_prem)

            # Per-trade bar chart
            if _raw_trials:
                import plotly.graph_objects as go
                _bar_pls   = []
                _bar_dates = []
                for _t in _raw_trials[:200]:
                    _t_pl  = float(_t.get("profitLoss") or _t.get("profit_loss") or 0)
                    _t_dt  = str(_t.get("openDateTime") or _t.get("entryDate") or _t.get("entry_date") or "")[:10]
                    _bar_pls.append(_t_pl)
                    _bar_dates.append(_t_dt or f"Trade {len(_bar_dates)+1}")

                _bar_colors = ["#10B981" if p >= 0 else "#EF4444" for p in _bar_pls]
                fig2 = go.Figure(go.Bar(
                    x=_bar_dates, y=_bar_pls,
                    marker_color=_bar_colors,
                    name="Profit/loss per trade",
                ))
                fig2.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0D1B2A",
                    plot_bgcolor="#0D1B2A",
                    title=dict(text="Profit / loss for all trades", font=dict(size=12, color="#94A3B8")),
                    margin=dict(l=50, r=20, t=40, b=40),
                    height=240,
                    yaxis=dict(tickprefix="$", tickfont=dict(size=10)),
                    xaxis=dict(tickfont=dict(size=9)),
                    showlegend=False,
                )
                st.plotly_chart(fig2, use_container_width=True)

            # Stats grid in 3 columns
            def _stat_row(label, value, color="#F1F5F9"):
                return (
                    f'<div style="color:#94A3B8;font-size:.77rem">{label}</div>'
                    f'<div style="color:{color};font-weight:700;font-size:.82rem">{value}</div>'
                )

            dc1, dc2, dc3 = st.columns(3)
            with dc1:
                st.markdown(
                    f'<div style="background:#0A2540;border:1px solid #1E4D7A;border-radius:8px;padding:.9rem 1rem">'
                    f'<div style="display:grid;grid-template-columns:auto 1fr;gap:.3rem .7rem;align-items:center">'
                    + _stat_row("Number of trades",        str(_n_tr))
                    + _stat_row("Trades with profits",     str(_n_wins), "#10B981")
                    + _stat_row("Profit rate",             f"{_pr_rate:.0f}%", "#10B981" if _pr_rate > 0 else "#94A3B8")
                    + _stat_row("Largest individual profit", f"${_lg_prof:+,.2f}", "#10B981" if _lg_prof >= 0 else "#EF4444")
                    + _stat_row("Trades with losses",      str(_n_loss), "#EF4444")
                    + _stat_row("Loss rate",               f"{_lr_rate:.0f}%", "#EF4444" if _lr_rate > 0 else "#94A3B8")
                    + _stat_row("Largest individual loss", f"${_lg_loss:+,.2f}", "#EF4444" if _lg_loss < 0 else "#94A3B8")
                    + f'</div></div>',
                    unsafe_allow_html=True,
                )
            with dc2:
                _rp_color  = "#10B981" if _avg_ret_pct >= 0 else "#EF4444"
                _pnl_color = "#10B981" if _avg_pnl_dollar >= 0 else "#EF4444"
                # Premium capture rate: only meaningful for SELL strategies
                _det_direction = (st.session_state.get("opts_params") or {}).get("direction", "Sell")
                _is_sell_det = _det_direction.lower() == "sell"
                _prem_cap_pct = (float(opts.get("profit_loss") or 0) / _tot_prem * 100) if (_tot_prem and _is_sell_det) else 0
                _prem_cap_color = "#10B981" if _prem_cap_pct >= 0 else "#EF4444"
                st.markdown(
                    f'<div style="background:#0A2540;border:1px solid #1E4D7A;border-radius:8px;padding:.9rem 1rem">'
                    f'<div style="display:grid;grid-template-columns:auto 1fr;gap:.3rem .7rem;align-items:center">'
                    + _stat_row("Avg. return per trade",      f"{_avg_ret_pct:+.2f}%" if _avg_ret_pct else "---", _rp_color)
                    + _stat_row("Avg. days in trade",         f"{_avg_days:.1f}" if _avg_days else "1")
                    + _stat_row("Avg. BPR per trade",         f"${_avg_bpr:,.2f}" if _avg_bpr else "---")
                    + _stat_row("Avg. premium",               f"${_avg_prem:+,.2f}" if _avg_prem else "---")
                    + _stat_row("Avg. profit/loss per trade", f"${_avg_pnl_dollar:+,.2f}" if _avg_pnl_dollar else "---", _pnl_color)
                    + _stat_row("Avg. win size",              f"${_avg_win:+,.2f}" if _avg_win else "$0")
                    + _stat_row("Avg. loss size",             f"${_avg_ls:+,.2f}" if _avg_ls else "---", "#EF4444")
                    + (_stat_row("Premium capture rate", f"{_prem_cap_pct:+,.2f}%", _prem_cap_color) if _is_sell_det else _stat_row("Premium capture rate", "N/A (Buy strategy)", "#64748B"))
                    + f'</div></div>',
                    unsafe_allow_html=True,
                )
            with dc3:
                _pnl_v2   = float(opts.get("profit_loss") or 0)
                _roc2     = (_pnl_v2 / _used_c * 100) if _used_c else 0
                _pnl_c    = "#10B981" if _pnl_v2 >= 0 else "#EF4444"
                _roc_c    = "#10B981" if _roc2 >= 0 else "#EF4444"
                st.markdown(
                    f'<div style="background:#0A2540;border:1px solid #1E4D7A;border-radius:8px;padding:.9rem 1rem">'
                    f'<div style="display:grid;grid-template-columns:auto 1fr;gap:.3rem .7rem;align-items:center">'
                    + _stat_row("Total profit/loss",        f"${_pnl_v2:+,.2f}", _pnl_c)
                    + _stat_row("Used capital",             f"${_used_c:,.2f}")
                    + _stat_row("Return on used capital",   f"{_roc2:.1f}%", _roc_c)
                    + _stat_row("CAGR",                     f"{_cagr_v:.1f}%" if _cagr_v else "---")
                    + _stat_row("MAR ratio",                f"{_mar:.2f}" if _mar else "---")
                    + _stat_row("Max drawdown",             f"{_raw_max_dd:.2f}%", "#EF4444")
                    + _stat_row("Total premium",            f"${_tot_prem:+,.2f}" if _tot_prem else "---")
                    + _stat_row("Total fees",               f"${_tot_fees:.2f}" if _tot_fees else "---")
                    + f'</div></div>',
                    unsafe_allow_html=True,
                )

        # ── AGENT TRADE MAP TAB ──────────────────────────────────────────────
        with _tt_tab_agent:
            _atm_ai_dec   = ai.get("decision", "---")
            _atm_ai_ret   = float(ai.get("predicted_return_pct") or 0)
            _atm_ai_conf  = ai.get("confidence_score", "?")
            _atm_dir      = opts_p.get("direction", "Buy").lower()
            _atm_type     = opts_p.get("opt_type", "Call").lower()
            _atm_strategy = f"{opts_p.get('direction','Buy')} {opts_p.get('opt_type','Call')}"
            # Strategy category flags — used in both verdict and entry logic below
            _is_sell_strategy = _atm_dir == "sell"
            _is_buy_strategy  = _atm_dir == "buy"

            # ── AGENT vs BACKTESTER COMPARISON HEADER ────────────────────────────
            _rtc = ai.get("recommended_trade_config") or {}
            _rtc_action     = _rtc.get("action") or "—"
            _rtc_delta      = _rtc.get("suggested_delta") or _rtc.get("suggested_delta_range") or opts_p.get("delta") or "—"
            _rtc_dte        = _rtc.get("suggested_dte") or _rtc.get("suggested_dte_range") or opts_p.get("dte") or "—"
            _rtc_qty        = _rtc.get("suggested_quantity") or opts_p.get("quantity") or "—"
            _rtc_align      = _rtc.get("alignment_with_user_config") or "—"
            _rtc_notes      = _rtc.get("alignment_notes") or ""

            # When AI and user are ALIGNED, the recommended action must match the
            # user's call/put type. Gemini often says "Sell Put" even when user
            # selected "Buy Call" (both bullish — so aligned, but different type).
            # Showing "Sell Put" when user picked "Buy Call" looks like a contradiction.
            # Fix: when ALIGNED, display user's own strategy as the recommendation.
            if _rtc_align == "ALIGNED":
                _rtc_action = _atm_strategy  # e.g. "Buy Call", "Sell Put"

            # Backtester totals
            _bt_total_trades = int(opts.get("total_trades") or len(_raw_trials or []) or 0)
            _bt_pnl          = float(opts.get("profit_loss") or 0)
            _bt_wr_raw       = float(opts.get("win_rate") or 0)
            _bt_wr_pct       = _bt_wr_raw * 100 if _bt_wr_raw <= 1.0 else _bt_wr_raw
            _bt_pnl_clr      = "#10B981" if _bt_pnl >= 0 else "#EF4444"
            _bt_wr_clr       = "#10B981" if _bt_wr_pct >= 60 else ("#F59E0B" if _bt_wr_pct >= 40 else "#EF4444")

            # Direction alignment verdict — strategy-aware:
            # • Buy Call: profitable → market went up → AI direction was right
            # • Buy Put: profitable → market went down → AI direction was right
            # • Sell Put: profitable → market didn't crash → HOLD or BUY was right
            # • Sell Call: profitable → market didn't surge → HOLD or SELL was right
            _dec_upper               = str(_atm_ai_dec).upper()
            _is_sell_strategy_verdict = _is_sell_strategy  # alias for clarity

            if _is_sell_strategy_verdict:
                # For premium sellers, direction is "correct" when strategy is profitable
                # AND the agent would have entered (HOLD or aligned direction)
                _direction_correct = _bt_pnl > 0 and not (_dec_upper == "REVIEW")
                # Also "correct" if agent chose not to enter (SELL dec for Sell Put) and would have saved from loss
                if not _direction_correct and _dec_upper == "SELL" and _atm_type == "put":
                    _direction_correct = _bt_pnl < 0  # agent was right to skip
            else:
                # Buy Call: needs BUY decision + positive P&L
                # Buy Put: needs SELL decision + positive P&L
                _direction_correct = (
                    (_atm_type == "call" and _dec_upper == "BUY" and _bt_pnl > 0) or
                    (_atm_type == "put"  and _dec_upper == "SELL" and _bt_pnl > 0)
                )
            _verdict_label  = "AGENT DIRECTION: CORRECT ✅" if _direction_correct else "AGENT DIRECTION: WRONG ❌"
            _verdict_color  = "#10B981" if _direction_correct else "#EF4444"
            _align_badge_color = (
                "#10B981" if _rtc_align == "ALIGNED"
                else "#F59E0B" if _rtc_align == "PARTIALLY_ALIGNED"
                else "#EF4444" if _rtc_align == "NOT_ALIGNED"
                else "#94A3B8"
            )
            _rtc_notes_html = (
                f'<div style="color:#94A3B8;font-size:.68rem;margin-top:.3rem">{_rtc_notes}</div>'
                if _rtc_notes else ""
            )

            # ── Final action verdict: TRADE / MODIFY / NO TRADE / REVIEW ──────
            # Computed early by replicating agent-entry logic (full logic at ~line 5014+)
            _ai_dec_early    = str(_atm_ai_dec).upper()
            _early_is_buy    = _ai_dec_early == "BUY"
            _early_is_sell   = _ai_dec_early == "SELL"
            _early_is_review = _ai_dec_early == "REVIEW"
            if _atm_dir == "buy" and _atm_type == "call":
                _agent_enters_pre = _early_is_buy and not _early_is_review
            elif _atm_dir == "buy" and _atm_type == "put":
                _agent_enters_pre = _early_is_sell and not _early_is_review
            elif _atm_dir == "sell" and _atm_type == "put":
                _agent_enters_pre = not _early_is_sell and not _early_is_review
            elif _atm_dir == "sell" and _atm_type == "call":
                _agent_enters_pre = not _early_is_buy and not _early_is_review
            else:
                _agent_enters_pre = not _early_is_review

            if _early_is_review:
                _final_verdict       = "REVIEW"
                _final_verdict_color = "#94A3B8"
                _final_verdict_icon  = "REVIEW"
                _final_verdict_desc  = "Data quality or confidence too low — do not trade without further review"
            elif not _agent_enters_pre:
                _final_verdict       = "AI CONFLICTS"
                _final_verdict_color = "#F97316"
                _final_verdict_icon  = "AI CONFLICTS"
                _final_verdict_desc  = f"AI signal ({_ai_dec_early}) conflicts with your strategy — see AI accuracy table below to decide"
            elif _rtc_align in ("NOT_ALIGNED", "PARTIALLY_ALIGNED"):
                _final_verdict       = "MODIFY"
                _final_verdict_color = "#F59E0B"
                _final_verdict_icon  = "MODIFY"
                _final_verdict_desc  = "AI recommends different parameters — adjust before entering"
            else:
                _final_verdict       = "TRADE"
                _final_verdict_color = "#10B981"
                _final_verdict_icon  = "TRADE"
                _final_verdict_desc  = "Strategy aligned with AI direction — enter the trade"

            # ── NEW: USER STRATEGY EVALUATION + AI RECOMMENDATION ─────────────
            _use = ai.get("user_strategy_evaluation") or {}
            _use_verdict   = str(_use.get("verdict") or "REVIEW").upper()
            _use_issues    = _use.get("issues") or []
            _use_strengths = _use.get("strengths") or []
            _use_assess    = _use.get("assessment") or ""

            # Verdict colors
            _uv_color = (
                "#10B981" if _use_verdict == "GOOD"
                else "#EF4444" if _use_verdict == "NOT_RECOMMENDED"
                else "#F59E0B" if _use_verdict == "RISKY"
                else "#94A3B8"
            )
            _uv_icon = (
                "✅ GOOD" if _use_verdict == "GOOD"
                else "❌ NOT RECOMMENDED" if _use_verdict == "NOT_RECOMMENDED"
                else "⚠️ RISKY" if _use_verdict == "RISKY"
                else "🔍 REVIEW"
            )

            # Build issues/strengths HTML
            _issues_html = "".join(
                f'<div style="color:#EF4444;font-size:.72rem;margin:.15rem 0">• {iss}</div>'
                for iss in _use_issues if iss
            )
            _strengths_html = "".join(
                f'<div style="color:#10B981;font-size:.72rem;margin:.15rem 0">• {s}</div>'
                for s in _use_strengths if s
            )

            # AI recommended strategy display
            _rtc_tp  = _rtc.get("suggested_take_profit_pct") or opts_p.get("take_profit_pct") or "—"
            _rtc_sl  = _rtc.get("suggested_stop_loss_pct")  or opts_p.get("stop_loss_pct")  or "—"
            _align_changed = _rtc_align in ("NOT_ALIGNED", "PARTIALLY_ALIGNED")

            st.markdown(
                f'<div style="background:linear-gradient(135deg,#0D0D20 0%,#0A0E1A 100%);'
                f'border:2px solid #312E81;border-radius:12px;padding:1.1rem 1.3rem;margin:.3rem 0 .8rem 0">'
                f'<div style="color:#A78BFA;font-size:.95rem;font-weight:900;letter-spacing:.05em;'
                f'margin-bottom:.9rem;border-bottom:1px solid #312E81;padding-bottom:.5rem">'
                f'🧠 AI STRATEGY INTELLIGENCE ENGINE — EVALUATE &amp; IMPROVE</div>'
                # Row: USER ENTERED vs AI EVALUATION
                f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:.8rem;margin-bottom:.8rem">'
                # LEFT: User's entered strategy
                f'<div style="background:#111827;border-radius:8px;padding:.8rem 1rem;border-left:3px solid #4B5563">'
                f'<div style="color:#64748B;font-size:.68rem;font-weight:700;letter-spacing:.05em;margin-bottom:.5rem">USER ENTERED STRATEGY</div>'
                f'<div style="color:#F1F5F9;font-size:.82rem;font-weight:800">{opts_p.get("direction","Buy").upper()} {opts_p.get("opt_type","Call").upper()}</div>'
                f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:.2rem .5rem;margin-top:.4rem">'
                f'<div style="color:#64748B;font-size:.68rem">Symbol</div><div style="color:#F1F5F9;font-size:.72rem;font-weight:700">{symbol}</div>'
                f'<div style="color:#64748B;font-size:.68rem">Delta</div><div style="color:#F1F5F9;font-size:.72rem;font-weight:700">{opts_p.get("delta","—")}</div>'
                f'<div style="color:#64748B;font-size:.68rem">DTE</div><div style="color:#F1F5F9;font-size:.72rem;font-weight:700">{opts_p.get("dte","—")}d</div>'
                f'<div style="color:#64748B;font-size:.68rem">Quantity</div><div style="color:#F1F5F9;font-size:.72rem;font-weight:700">{opts_p.get("quantity","—")} contracts</div>'
                f'<div style="color:#64748B;font-size:.68rem">Take Profit</div><div style="color:#F1F5F9;font-size:.72rem;font-weight:700">{opts_p.get("take_profit_pct","—")}%</div>'
                f'<div style="color:#64748B;font-size:.68rem">Stop Loss</div><div style="color:#F1F5F9;font-size:.72rem;font-weight:700">{opts_p.get("stop_loss_pct","—")}%</div>'
                f'</div></div>'
                # RIGHT: AI evaluation
                f'<div style="background:#111827;border-radius:8px;padding:.8rem 1rem;border-left:3px solid {_uv_color}">'
                f'<div style="color:#64748B;font-size:.68rem;font-weight:700;letter-spacing:.05em;margin-bottom:.5rem">AI EVALUATION OF YOUR STRATEGY</div>'
                f'<div style="color:{_uv_color};font-size:.85rem;font-weight:900;margin-bottom:.4rem">{_uv_icon}</div>'
                + (f'<div style="color:#94A3B8;font-size:.7rem;margin-bottom:.3rem">{_use_assess}</div>' if _use_assess else "")
                + (_issues_html if _issues_html else "")
                + (_strengths_html if _strengths_html else "")
                + f'</div>'
                f'</div>'
                # AI RECOMMENDED STRATEGY (always show)
                f'<div style="background:{"#0A1F0A" if _rtc_align=="ALIGNED" else "#1A0A20"};'
                f'border:1px solid {"#166534" if _rtc_align=="ALIGNED" else "#581C87"};'
                f'border-radius:8px;padding:.8rem 1rem;margin-bottom:.8rem">'
                f'<div style="color:#64748B;font-size:.68rem;font-weight:700;letter-spacing:.05em;margin-bottom:.5rem">'
                f'AI RECOMMENDED STRATEGY {"(SAME AS YOURS)" if _rtc_align=="ALIGNED" else "(DIFFERENT FROM YOURS — CONSIDER SWITCHING)"}</div>'
                f'<div style="display:grid;grid-template-columns:repeat(4,1fr) repeat(3,1fr);gap:.3rem .6rem">'
                f'<div><div style="color:#64748B;font-size:.63rem">ACTION</div>'
                f'<div style="color:#A78BFA;font-size:.78rem;font-weight:800">{_rtc_action}</div></div>'
                f'<div><div style="color:#64748B;font-size:.63rem">DELTA</div>'
                f'<div style="color:#60A5FA;font-size:.78rem;font-weight:800">{_rtc_delta}</div></div>'
                f'<div><div style="color:#64748B;font-size:.63rem">DTE</div>'
                f'<div style="color:#60A5FA;font-size:.78rem;font-weight:800">{_rtc_dte}d</div></div>'
                f'<div><div style="color:#64748B;font-size:.63rem">QTY</div>'
                f'<div style="color:#60A5FA;font-size:.78rem;font-weight:800">{_rtc_qty}</div></div>'
                f'<div><div style="color:#64748B;font-size:.63rem">TP%</div>'
                f'<div style="color:#10B981;font-size:.78rem;font-weight:800">{_rtc_tp}%</div></div>'
                f'<div><div style="color:#64748B;font-size:.63rem">SL%</div>'
                f'<div style="color:#F59E0B;font-size:.78rem;font-weight:800">{_rtc_sl}%</div></div>'
                f'<div><div style="color:#64748B;font-size:.63rem">ALIGNMENT</div>'
                f'<div style="color:{_align_badge_color};font-size:.7rem;font-weight:800">{_rtc_align}</div></div>'
                f'</div>'
                + (f'<div style="color:#94A3B8;font-size:.7rem;margin-top:.4rem">{_rtc_notes}</div>' if _rtc_notes else "")
                + f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # ── WHAT AI LEARNED FROM CONTEXT PERIOD (TT Historical Intelligence) ──
            _ctx_opt = st.session_state.get("ctx_optimizer_results") or {}
            _ctx_res = _ctx_opt.get("results") or []
            if _ctx_res:
                _ctx_valid = [r for r in _ctx_res if not r.get("no_data")]
                _ctx_total = _ctx_opt.get("total_combos", len(_ctx_res))
                _ctx_valid_n = _ctx_opt.get("valid_combos", len(_ctx_valid))
                _ctx_dir_lbl = "BUY" if _ctx_opt.get("direction", "long") == "long" else "SELL"
                _ctx_from = _ctx_opt.get("ctx_start", "")
                _ctx_to   = _ctx_opt.get("origin_date", "")
                _ctx_iv   = _ctx_opt.get("current_iv")

                st.markdown(
                    f'<div style="background:linear-gradient(135deg,#030B1A 0%,#071426 100%);'
                    f'border:2px solid #3B82F6;border-radius:12px;padding:.8rem 1.2rem;margin:.6rem 0 .4rem 0">'
                    f'<div style="color:#60A5FA;font-size:.88rem;font-weight:900;letter-spacing:.05em">'
                    f'🧠 AI LEARNING INTELLIGENCE — What AI studied before predicting</div>'
                    f'<div style="color:#64748B;font-size:.72rem;margin-top:.2rem">'
                    f'Tested {_ctx_total} real TastyTrade backtests on the context period '
                    f'({_ctx_from} → {_ctx_to}). '
                    f'{_ctx_valid_n} returned data. AI used these results to recommend the best parameters '
                    f'— not just stock charts.</div></div>',
                    unsafe_allow_html=True,
                )

                if _ctx_valid:
                    _ctx_best = _ctx_valid[0]
                    _ctx_worst = _ctx_valid[-1]
                    _ctx_iv_str = (
                        f"{_ctx_iv*100:.1f}% IV"
                        if _ctx_iv else "IV: N/A"
                    )
                    _ctx_iv_label = (
                        "HIGH — options expensive" if (_ctx_iv or 0) > 0.35
                        else "LOW — options cheap" if (_ctx_iv or 0) < 0.18
                        else "MODERATE"
                    ) if _ctx_iv else ""

                    # Summary cards
                    _cxa, _cxb, _cxc, _cxd = st.columns(4)
                    with _cxa:
                        st.markdown(
                            f'<div style="background:#0A1F0A;border:1px solid #10B981;border-radius:8px;padding:.6rem;text-align:center">'
                            f'<div style="color:#64748B;font-size:.62rem;font-weight:700">BEST COMBO FOUND</div>'
                            f'<div style="color:#10B981;font-size:.82rem;font-weight:900">D{_ctx_best["delta"]} · {_ctx_best["dte"]}DTE</div>'
                            f'<div style="color:#F59E0B;font-size:.72rem">TP{_ctx_best["tp_pct"]}% · SL{_ctx_best["sl_pct"]}%</div></div>',
                            unsafe_allow_html=True,
                        )
                    with _cxb:
                        _cx_pclr = "#10B981" if _ctx_best["total_pnl"] >= 0 else "#EF4444"
                        st.markdown(
                            f'<div style="background:#0A1F0A;border:1px solid #10B981;border-radius:8px;padding:.6rem;text-align:center">'
                            f'<div style="color:#64748B;font-size:.62rem;font-weight:700">BEST CONTEXT P&L</div>'
                            f'<div style="color:{_cx_pclr};font-size:.9rem;font-weight:900">${_ctx_best["total_pnl"]:+,.0f}</div>'
                            f'<div style="color:#94A3B8;font-size:.72rem">{_ctx_best["win_rate"]*100:.0f}% win · {_ctx_best["trades"]} trades</div></div>',
                            unsafe_allow_html=True,
                        )
                    with _cxc:
                        _cw_pclr = "#EF4444" if _ctx_worst["total_pnl"] < 0 else "#F59E0B"
                        st.markdown(
                            f'<div style="background:#200A0A;border:1px solid #EF4444;border-radius:8px;padding:.6rem;text-align:center">'
                            f'<div style="color:#64748B;font-size:.62rem;font-weight:700">WORST COMBO</div>'
                            f'<div style="color:{_cw_pclr};font-size:.82rem;font-weight:900">D{_ctx_worst["delta"]} · {_ctx_worst["dte"]}DTE</div>'
                            f'<div style="color:#EF4444;font-size:.72rem">${_ctx_worst["total_pnl"]:+,.0f} | avoid this</div></div>',
                            unsafe_allow_html=True,
                        )
                    with _cxd:
                        st.markdown(
                            f'<div style="background:#0A0F1E;border:1px solid #3B82F6;border-radius:8px;padding:.6rem;text-align:center">'
                            f'<div style="color:#64748B;font-size:.62rem;font-weight:700">CURRENT IV</div>'
                            f'<div style="color:#60A5FA;font-size:.9rem;font-weight:900">{_ctx_iv_str}</div>'
                            f'<div style="color:#94A3B8;font-size:.72rem">{_ctx_iv_label}</div></div>',
                            unsafe_allow_html=True,
                        )

                    # Context-period ranked table
                    _ctx_tbl = (
                        '<div style="overflow-x:auto;margin-top:.6rem">'
                        '<table style="width:100%;border-collapse:collapse;font-size:.72rem;font-family:monospace">'
                        '<thead><tr style="background:#0F172A;border-bottom:2px solid #3B82F6">'
                        '<th style="padding:.38rem .5rem;color:#60A5FA;text-align:center">RANK</th>'
                        '<th style="padding:.38rem .5rem;color:#60A5FA;text-align:center">DELTA</th>'
                        '<th style="padding:.38rem .5rem;color:#60A5FA;text-align:center">DTE</th>'
                        '<th style="padding:.38rem .5rem;color:#10B981;text-align:center">TP%</th>'
                        '<th style="padding:.38rem .5rem;color:#F59E0B;text-align:center">SL%</th>'
                        '<th style="padding:.38rem .5rem;color:#60A5FA;text-align:right">CTX P&L</th>'
                        '<th style="padding:.38rem .5rem;color:#60A5FA;text-align:right">WIN RATE</th>'
                        '<th style="padding:.38rem .5rem;color:#60A5FA;text-align:right">TRADES</th>'
                        '<th style="padding:.38rem .5rem;color:#60A5FA;text-align:right">AVG/TRADE</th>'
                        '</tr></thead><tbody>'
                    )
                    _ctx_rank_idx = 0
                    _ctx_valid_list = [r for r in _ctx_res if not r.get("no_data")]
                    for _cxi, _cxrow in enumerate(_ctx_res):
                        _cx_nd = _cxrow.get("no_data", False)
                        if not _cx_nd:
                            _ctx_rank_idx += 1
                        _cx_best_r  = not _cx_nd and _ctx_rank_idx == 1
                        _cx_worst_r = not _cx_nd and _ctx_rank_idx == len(_ctx_valid_list)
                        _cx_bg  = "#061A06" if _cx_best_r else ("#1A0606" if _cx_worst_r else ("#0A0A0A" if _cx_nd else "#060E1A"))
                        _cx_bdr = "border-left:3px solid #10B981;" if _cx_best_r else (
                            "border-left:3px solid #EF4444;" if _cx_worst_r else
                            "border-left:3px solid #374151;" if _cx_nd else
                            "border-left:3px solid #3B82F6;"
                        )
                        _cx_rnk = ("🥇 BEST" if _cx_best_r else
                                   f"💀 WORST" if _cx_worst_r else
                                   "⚫ no data" if _cx_nd else f"#{_ctx_rank_idx}")
                        _cx_pnl  = "—" if _cx_nd else "${:+,.0f}".format(_cxrow["total_pnl"])
                        _cx_wr   = "—" if _cx_nd else "{:.0f}%".format(_cxrow["win_rate"]*100)
                        _cx_tr   = "—" if _cx_nd else str(_cxrow["trades"])
                        _cx_avg  = "—" if _cx_nd else "${:+,.0f}".format(_cxrow.get("avg_pnl", 0))
                        _cx_pc   = ("#64748B" if _cx_nd else
                                    "#10B981" if _cxrow["total_pnl"] >= 0 else "#EF4444")
                        _cx_wc   = "#64748B" if _cx_nd else (
                            "#10B981" if _cxrow["win_rate"] >= 0.6 else
                            "#F59E0B" if _cxrow["win_rate"] >= 0.4 else "#EF4444")
                        _cx_op   = "opacity:0.4;" if _cx_nd else ""
                        _ctx_tbl += (
                            f'<tr style="background:{_cx_bg};{_cx_bdr}border-bottom:1px solid #1E3A5F;{_cx_op}">'
                            f'<td style="padding:.32rem .5rem;text-align:center;color:#60A5FA">{_cx_rnk}</td>'
                            f'<td style="padding:.32rem .5rem;text-align:center;color:#F1F5F9;font-weight:700">{_cxrow["delta"]}</td>'
                            f'<td style="padding:.32rem .5rem;text-align:center;color:#F1F5F9">{_cxrow["dte"]}d</td>'
                            f'<td style="padding:.32rem .5rem;text-align:center;color:#10B981;font-weight:700">{_cxrow["tp_pct"]}%</td>'
                            f'<td style="padding:.32rem .5rem;text-align:center;color:#F59E0B;font-weight:700">{_cxrow["sl_pct"]}%</td>'
                            f'<td style="padding:.32rem .5rem;text-align:right;color:{_cx_pc};font-weight:800">{_cx_pnl}</td>'
                            f'<td style="padding:.32rem .5rem;text-align:right;color:{_cx_wc};font-weight:700">{_cx_wr}</td>'
                            f'<td style="padding:.32rem .5rem;text-align:right;color:#CBD5E1">{_cx_tr}</td>'
                            f'<td style="padding:.32rem .5rem;text-align:right;color:{_cx_pc}">{_cx_avg}</td>'
                            f'</tr>'
                        )
                    _ctx_tbl += '</tbody></table></div>'
                    st.markdown(_ctx_tbl, unsafe_allow_html=True)

                    # AI Recommendation vs Context Best comparison
                    _ai_rec_act_cx = (ai.get("recommended_trade_config") or {})
                    _ai_rec_d_cx   = int(_ai_rec_act_cx.get("suggested_delta") or 0)
                    _ai_rec_dte_cx = int(_ai_rec_act_cx.get("suggested_dte") or 0)
                    if _ai_rec_d_cx and _ai_rec_dte_cx:
                        _cx_match = (
                            abs(_ai_rec_d_cx - _ctx_best["delta"]) <= 10 and
                            abs(_ai_rec_dte_cx - _ctx_best["dte"]) <= 7
                        )
                        st.markdown(
                            f'<div style="background:#0A0F1E;border:1px solid #3B82F6;border-radius:8px;'
                            f'padding:.6rem 1rem;margin-top:.5rem;font-size:.74rem">'
                            f'<span style="color:#60A5FA;font-weight:900">🤖 AI LEARNED: </span>'
                            f'<span style="color:#F1F5F9">AI recommends D{_ai_rec_d_cx} · {_ai_rec_dte_cx}DTE based on context intelligence. '
                            f'Context best was D{_ctx_best["delta"]} · {_ctx_best["dte"]}DTE. </span>'
                            f'<span style="color:{"#10B981" if _cx_match else "#F59E0B"};font-weight:700">'
                            f'{"✅ AI aligned with historical best." if _cx_match else "⚠️ AI deviated — check AI rationale."}'
                            f'</span></div>',
                            unsafe_allow_html=True,
                        )

            # ── STRATEGY OPTIMIZER DISPLAY (auto-ran in computation block) ──────
            _sopt_display = opts.get("optimizer_results") or []
            _sopt_has_results = bool(_sopt_display)
            if _sopt_display:
                _sopt_tested    = opts.get("optimizer_total_tested", len(_sopt_display))
                _sopt_valid_n   = sum(1 for r in _sopt_display if not r.get("no_data"))
                _sopt_dir_label = "SELL" if opts_p.get("direction","").lower() in ("sell","short") else "BUY"
                st.markdown(
                    f'<div style="background:linear-gradient(135deg,#0A0A05 0%,#14140A 100%);'
                    f'border:2px solid #D97706;border-radius:12px;padding:.8rem 1.2rem;margin:.6rem 0 .4rem 0">'
                    f'<div style="color:#F59E0B;font-size:.88rem;font-weight:900;letter-spacing:.05em">'
                    f'🔍 STRATEGY OPTIMIZER — tested {_sopt_tested} combos for {_sopt_dir_label} strategy, '
                    f'{_sopt_valid_n} returned data</div>'
                    f'<div style="color:#64748B;font-size:.72rem;margin-top:.2rem">'
                    f'Every combination of Delta · DTE · TP% · SL% below was run via real TastyTrade API '
                    f'on the same symbol &amp; date range. #1 BEST = the parameters that would have '
                    f'performed best historically.</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                # Build ranked table
                _sopt_tbl = (
                    '<div style="overflow-x:auto;border-radius:8px;border:1px solid #1E3A5F;margin-bottom:.8rem">'
                    '<table style="width:100%;border-collapse:collapse;font-size:.74rem">'
                    '<thead><tr style="background:#0D1A2E;color:#94A3B8;font-weight:700">'
                    '<th style="padding:.5rem .6rem;text-align:center">RANK</th>'
                    '<th style="padding:.5rem .6rem;text-align:center">DELTA</th>'
                    '<th style="padding:.5rem .6rem;text-align:center">DTE</th>'
                    '<th style="padding:.5rem .6rem;text-align:center;color:#10B981">TP%</th>'
                    '<th style="padding:.5rem .6rem;text-align:center;color:#EF4444">SL%</th>'
                    '<th style="padding:.5rem .6rem;text-align:right">TOTAL P&L</th>'
                    '<th style="padding:.5rem .6rem;text-align:right">WIN RATE</th>'
                    '<th style="padding:.5rem .6rem;text-align:right">TRADES</th>'
                    '<th style="padding:.5rem .6rem;text-align:right">AVG/TRADE</th>'
                    '</tr></thead><tbody>'
                )
                _sopt_valid_rows = [r for r in _sopt_display if not r.get("no_data")]
                _sopt_rank_idx = 0
                for _sri, _srow in enumerate(_sopt_display):
                    _s_nodata = _srow.get("no_data", False)
                    if not _s_nodata:
                        _sopt_rank_idx += 1
                    _s_best  = not _s_nodata and _sopt_rank_idx == 1
                    _s_worst = not _s_nodata and _sopt_rank_idx == len(_sopt_valid_rows)
                    _s_bg    = "#0A1F0A" if _s_best else ("#200A0A" if _s_worst else ("#111111" if _s_nodata else "#0A0F1E"))
                    _s_bdr   = (
                        "border-left:3px solid #10B981;" if _s_best
                        else "border-left:3px solid #EF4444;" if _s_worst
                        else "border-left:3px solid #374151;" if _s_nodata
                        else "border-left:3px solid #1E3A5F;"
                    )
                    _s_pclr = "#10B981" if _srow["total_pnl"] >= 0 else "#EF4444"
                    _s_wrp  = _srow["win_rate"] * 100
                    _s_wclr = "#10B981" if _s_wrp >= 60 else ("#F59E0B" if _s_wrp >= 40 else "#EF4444")
                    _s_aclr = "#10B981" if _srow["avg_pnl"] >= 0 else "#EF4444"
                    if _s_nodata:
                        _s_rank = "⚫ no data"
                    elif _s_best:
                        _s_rank = "🥇 #1 BEST"
                    elif _s_worst:
                        _s_rank = f"💀 #{_sopt_rank_idx} WORST"
                    else:
                        _s_rank = f"#{_sopt_rank_idx}"
                    # Pre-compute all display values to keep f-strings readable
                    _s_nd_op   = "opacity:0.45;" if _s_nodata else ""
                    _s_rk_clr  = "#F59E0B" if _s_best else ("#475569" if _s_nodata else "#94A3B8")
                    _s_rk_fw   = "900" if _s_best else "400"
                    _s_pnl_clr = "#475569" if _s_nodata else _s_pclr
                    _s_wr_clr2 = "#475569" if _s_nodata else _s_wclr
                    _s_tr_clr  = "#475569" if _s_nodata else "#CBD5E1"
                    _s_av_clr  = "#475569" if _s_nodata else _s_aclr
                    _s_pnl_str = "—" if _s_nodata else "${:+,.0f}".format(_srow["total_pnl"])
                    _s_wr_str  = "—" if _s_nodata else "{:.0f}%".format(_s_wrp)
                    _s_tr_str  = "—" if _s_nodata else str(_srow["trades"])
                    _s_av_str  = "—" if _s_nodata else "${:+,.0f}".format(_srow["avg_pnl"])
                    _sopt_tbl += (
                        f'<tr style="background:{_s_bg};{_s_bdr}border-bottom:1px solid #1E3A5F;{_s_nd_op}">'
                        f'<td style="padding:.42rem .6rem;text-align:center;color:{_s_rk_clr};font-weight:{_s_rk_fw}">{_s_rank}</td>'
                        f'<td style="padding:.42rem .6rem;text-align:center;color:#F1F5F9;font-weight:700">{_srow["delta"]}</td>'
                        f'<td style="padding:.42rem .6rem;text-align:center;color:#F1F5F9">{_srow["dte"]}d</td>'
                        f'<td style="padding:.42rem .6rem;text-align:center;color:#10B981;font-weight:700">{_srow["tp_pct"]}%</td>'
                        f'<td style="padding:.42rem .6rem;text-align:center;color:#F59E0B;font-weight:700">{_srow["sl_pct"]}%</td>'
                        f'<td style="padding:.42rem .6rem;text-align:right;color:{_s_pnl_clr};font-weight:800">{_s_pnl_str}</td>'
                        f'<td style="padding:.42rem .6rem;text-align:right;color:{_s_wr_clr2};font-weight:700">{_s_wr_str}</td>'
                        f'<td style="padding:.42rem .6rem;text-align:right;color:{_s_tr_clr}">{_s_tr_str}</td>'
                        f'<td style="padding:.42rem .6rem;text-align:right;color:{_s_av_clr}">{_s_av_str}</td>'
                        f'</tr>'
                    )
                _sopt_tbl += '</tbody></table></div>'
                st.markdown(_sopt_tbl, unsafe_allow_html=True)

                # Best vs User comparison
                _sob      = _sopt_display[0]
                _sob_pnl  = _sob["total_pnl"]
                _sob_wrp  = _sob["win_rate"] * 100
                _sob_n    = _sob["trades"]
                _sob_avg  = _sob["avg_pnl"]
                _sob_pclr = "#10B981" if _sob_pnl >= 0 else "#EF4444"
                _sob_wclr = "#10B981" if _sob_wrp >= 60 else ("#F59E0B" if _sob_wrp >= 40 else "#EF4444")
                _sob_diff = _sob_pnl - _bt_pnl
                _sob_wins = _sob_pnl > _bt_pnl
                _sob_tie  = abs(_sob_diff) < 1
                _sob_lbl  = "OPTIMIZER WINS" if _sob_wins else ("TIE" if _sob_tie else "YOUR STRATEGY WINS")
                _sob_clr  = "#F59E0B" if _sob_wins else ("#94A3B8" if _sob_tie else "#10B981")
                _sob_utp  = float(opts_p.get("take_profit_pct") or 0)
                _sob_usl  = float(opts_p.get("stop_loss_pct") or 0)
                _sob_bann = (
                    f"🥇 USE OPTIMIZER PARAMS — D{_sob['delta']} · {_sob['dte']}DTE · "
                    f"TP{_sob['tp_pct']}% · SL{_sob['sl_pct']}% performed better historically"
                    if _sob_wins else
                    "📌 YOUR STRATEGY IS NEAR-OPTIMAL — optimizer found no significantly better combo for this period"
                )
                st.markdown(
                    f'<div style="background:linear-gradient(135deg,#0A0F1E 0%,#0D1A2E 100%);'
                    f'border:2px solid #D97706;border-radius:12px;padding:1rem 1.2rem;margin:.3rem 0 .8rem 0">'
                    f'<div style="color:#F59E0B;font-size:.88rem;font-weight:900;letter-spacing:.05em;'
                    f'margin-bottom:.4rem;border-bottom:1px solid #2A2010;padding-bottom:.5rem">'
                    f'🏆 YOUR STRATEGY vs OPTIMIZER BEST — WHICH WINS?</div>'
                    f'<div style="display:grid;grid-template-columns:1fr auto 1fr;gap:.8rem;align-items:center">'
                    f'<div style="background:#111827;border-radius:8px;padding:.8rem;border-top:3px solid #4B5563">'
                    f'<div style="color:#94A3B8;font-size:.68rem;font-weight:700;margin-bottom:.3rem">YOUR STRATEGY</div>'
                    f'<div style="color:#F1F5F9;font-size:.75rem;font-weight:700">'
                    f'{opts_p.get("direction","—")} {opts_p.get("opt_type","—")} · '
                    f'D{opts_p.get("delta","—")} · {opts_p.get("dte","—")}DTE · {opts_p.get("quantity","—")}qty</div>'
                    f'<div style="color:#94A3B8;font-size:.63rem;margin-bottom:.4rem">TP {_sob_utp:.0f}% · SL {_sob_usl:.0f}%</div>'
                    f'<div style="color:#64748B;font-size:.63rem">TOTAL P&L</div>'
                    f'<div style="color:{_bt_pnl_clr};font-size:1.2rem;font-weight:900">${_bt_pnl:+,.0f}</div>'
                    f'<div style="display:flex;gap:.8rem;margin-top:.4rem">'
                    f'<div><div style="color:#64748B;font-size:.63rem">WIN RATE</div>'
                    f'<div style="color:{_bt_wr_clr};font-size:.85rem;font-weight:800">{_bt_wr_pct:.0f}%</div></div>'
                    f'<div><div style="color:#64748B;font-size:.63rem">TRADES</div>'
                    f'<div style="color:#F1F5F9;font-size:.85rem;font-weight:800">{_bt_total_trades}</div></div>'
                    f'<div><div style="color:#64748B;font-size:.63rem">AVG/TRADE</div>'
                    f'<div style="color:#F1F5F9;font-size:.85rem;font-weight:800">'
                    f'${(_bt_pnl / _bt_total_trades if _bt_total_trades else 0):+,.0f}</div></div>'
                    f'</div></div>'
                    f'<div style="text-align:center">'
                    f'<div style="color:{_sob_clr};font-size:1.5rem;font-weight:900">VS</div>'
                    f'<div style="color:{_sob_clr};font-size:.65rem;font-weight:800;margin-top:.2rem">{_sob_lbl}</div>'
                    f'<div style="color:#475569;font-size:.62rem;margin-top:.15rem">diff: ${_sob_diff:+,.0f}</div>'
                    f'</div>'
                    f'<div style="background:#111827;border-radius:8px;padding:.8rem;border-top:3px solid #D97706">'
                    f'<div style="color:#F59E0B;font-size:.68rem;font-weight:700;margin-bottom:.3rem">🥇 OPTIMIZER BEST</div>'
                    f'<div style="color:#F1F5F9;font-size:.75rem;font-weight:700">'
                    f'{opts_p.get("direction","—")} {opts_p.get("opt_type","—")} · '
                    f'D{_sob["delta"]} · {_sob["dte"]}DTE · {_sob["qty"]}qty</div>'
                    f'<div style="color:#94A3B8;font-size:.63rem;margin-bottom:.4rem">TP {_sob["tp_pct"]}% · SL {_sob["sl_pct"]}%</div>'
                    f'<div style="color:#64748B;font-size:.63rem">TOTAL P&L</div>'
                    f'<div style="color:{_sob_pclr};font-size:1.2rem;font-weight:900">${_sob_pnl:+,.0f}</div>'
                    f'<div style="display:flex;gap:.8rem;margin-top:.4rem">'
                    f'<div><div style="color:#64748B;font-size:.63rem">WIN RATE</div>'
                    f'<div style="color:{_sob_wclr};font-size:.85rem;font-weight:800">{_sob_wrp:.0f}%</div></div>'
                    f'<div><div style="color:#64748B;font-size:.63rem">TRADES</div>'
                    f'<div style="color:#F1F5F9;font-size:.85rem;font-weight:800">{_sob_n}</div></div>'
                    f'<div><div style="color:#64748B;font-size:.63rem">AVG/TRADE</div>'
                    f'<div style="color:#F1F5F9;font-size:.85rem;font-weight:800">${_sob_avg:+,.0f}</div></div>'
                    f'</div></div>'
                    f'</div>'
                    f'<div style="background:{"#0A1F0A" if _sob_wins else "#0A1430"};'
                    f'border-radius:6px;padding:.55rem .9rem;margin-top:.6rem;text-align:center">'
                    f'<span style="color:{_sob_clr};font-size:.82rem;font-weight:900">{_sob_bann}</span>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

            # ── AI STRATEGY BACKTEST COMPARISON (if AI ran a second backtest) ─
            # Suppressed when optimizer results are shown — optimizer comparison already
            # covers the "user vs best" angle, showing both would look like a duplicate.
            _ai_rec_res = opts.get("ai_rec_strategy_result") or {}
            _ai_str_res = opts.get("ai_strategy_result") or {}
            # Prefer the recommended-params result (real delta/DTE/qty/TP/SL from AI)
            # over the direction-only result (same params, different direction)
            _use_rec_res  = _ai_rec_res.get("status") == "SUCCESS"
            _active_ai    = _ai_rec_res if _use_rec_res else _ai_str_res
            if _active_ai.get("status") == "SUCCESS" and not _sopt_has_results:
                _ai_pnl   = float(_active_ai.get("profit_loss") or 0)
                _ai_wr_r  = float(_active_ai.get("win_rate") or 0)
                _ai_wr_p  = _ai_wr_r * 100 if _ai_wr_r <= 1.0 else _ai_wr_r
                _ai_n_tr  = int(_active_ai.get("total_trades") or 0)
                _ai_avg   = float(_active_ai.get("avg_pnl") or 0)
                _ai_better   = _ai_pnl > _bt_pnl
                _pnl_diff    = _ai_pnl - _bt_pnl
                _winner_label = "AI WINS" if _ai_better else ("TIE" if _pnl_diff == 0 else "YOUR STRATEGY WINS")
                _winner_color = "#A78BFA" if _ai_better else ("#F59E0B" if _pnl_diff == 0 else "#10B981")

                # Build AI side label
                _ai_rec_action_disp = str(_active_ai.get("action") or "—").title()
                _ai_rec_delta_disp  = _active_ai.get("delta", "—")
                _ai_rec_dte_disp    = _active_ai.get("dte", "—")
                _ai_rec_qty_disp    = _active_ai.get("quantity", "—")
                _ai_rec_tp_disp     = _active_ai.get("take_profit_pct")
                _ai_rec_sl_disp     = _active_ai.get("stop_loss_pct")
                _ai_rec_subtitle    = (
                    "AI's own recommended parameters — different Delta, DTE, Qty, TP, SL"
                    if _use_rec_res else
                    "Same parameters as yours — only direction flipped based on AI signal"
                )
                _ai_rec_tp_str = f"TP {_ai_rec_tp_disp:.0f}%" if _ai_rec_tp_disp else ""
                _ai_rec_sl_str = f"SL {_ai_rec_sl_disp:.0f}%" if _ai_rec_sl_disp else ""
                _ai_rec_exit   = " · ".join(filter(None, [_ai_rec_tp_str, _ai_rec_sl_str]))

                _user_tp_str = f"TP {float(opts_p.get('take_profit_pct') or 0):.0f}%" if opts_p.get("take_profit_pct") else ""
                _user_sl_str = f"SL {float(opts_p.get('stop_loss_pct') or 0):.0f}%" if opts_p.get("stop_loss_pct") else ""
                _user_exit   = " · ".join(filter(None, [_user_tp_str, _user_sl_str]))

                st.markdown(
                    f'<div style="background:linear-gradient(135deg,#0A0F1E 0%,#0D1A2E 100%);'
                    f'border:2px solid #1E3A5F;border-radius:12px;padding:1rem 1.2rem;margin:.3rem 0 1rem 0">'
                    f'<div style="color:#60A5FA;font-size:.88rem;font-weight:900;letter-spacing:.05em;'
                    f'margin-bottom:.4rem;border-bottom:1px solid #1E3A5F;padding-bottom:.5rem">'
                    f'📊 DUAL BACKTEST: YOUR STRATEGY vs AI RECOMMENDED STRATEGY — WHO WINS?</div>'
                    f'<div style="color:#64748B;font-size:.72rem;margin-bottom:.7rem">{_ai_rec_subtitle}</div>'
                    f'<div style="display:grid;grid-template-columns:1fr auto 1fr;gap:.8rem;align-items:center">'
                    # User strategy column
                    f'<div style="background:#111827;border-radius:8px;padding:.8rem;border-top:3px solid #4B5563">'
                    f'<div style="color:#94A3B8;font-size:.68rem;font-weight:700;margin-bottom:.4rem">YOUR STRATEGY</div>'
                    f'<div style="color:#F1F5F9;font-size:.75rem;font-weight:700">'
                    f'{opts_p.get("direction","—")} {opts_p.get("opt_type","—")} · D{opts_p.get("delta","—")} · {opts_p.get("dte","—")}DTE · {opts_p.get("quantity","—")}qty</div>'
                    f'<div style="color:#94A3B8;font-size:.63rem;margin-bottom:.3rem">{_user_exit}</div>'
                    f'<div style="margin-top:.4rem">'
                    f'<div style="color:#64748B;font-size:.63rem">TOTAL P&L</div>'
                    f'<div style="color:{_bt_pnl_clr};font-size:1.2rem;font-weight:900">${_bt_pnl:+,.0f}</div></div>'
                    f'<div style="display:flex;gap:.8rem;margin-top:.4rem">'
                    f'<div><div style="color:#64748B;font-size:.63rem">WIN RATE</div>'
                    f'<div style="color:{_bt_wr_clr};font-size:.85rem;font-weight:800">{_bt_wr_pct:.0f}%</div></div>'
                    f'<div><div style="color:#64748B;font-size:.63rem">TRADES</div>'
                    f'<div style="color:#F1F5F9;font-size:.85rem;font-weight:800">{_bt_total_trades}</div></div>'
                    f'<div><div style="color:#64748B;font-size:.63rem">AVG/TRADE</div>'
                    f'<div style="color:#F1F5F9;font-size:.85rem;font-weight:800">${(_bt_pnl/_bt_total_trades if _bt_total_trades else 0):+,.0f}</div></div>'
                    f'</div></div>'
                    # vs divider
                    f'<div style="text-align:center">'
                    f'<div style="color:{_winner_color};font-size:1.5rem;font-weight:900">VS</div>'
                    f'<div style="color:{_winner_color};font-size:.68rem;font-weight:800;margin-top:.2rem">{_winner_label}</div>'
                    f'<div style="color:#475569;font-size:.65rem;margin-top:.2rem">P&L diff: ${_pnl_diff:+,.0f}</div>'
                    f'</div>'
                    # AI strategy column
                    f'<div style="background:#111827;border-radius:8px;padding:.8rem;border-top:3px solid #A78BFA">'
                    f'<div style="color:#A78BFA;font-size:.68rem;font-weight:700;margin-bottom:.4rem">AI RECOMMENDED STRATEGY</div>'
                    f'<div style="color:#F1F5F9;font-size:.75rem;font-weight:700">'
                    f'{_ai_rec_action_disp} · D{_ai_rec_delta_disp} · {_ai_rec_dte_disp}DTE · {_ai_rec_qty_disp}qty</div>'
                    f'<div style="color:#94A3B8;font-size:.63rem;margin-bottom:.3rem">{_ai_rec_exit}</div>'
                    f'<div style="margin-top:.4rem">'
                    f'<div style="color:#64748B;font-size:.63rem">TOTAL P&L</div>'
                    f'<div style="color:{"#10B981" if _ai_pnl>=0 else "#EF4444"};font-size:1.2rem;font-weight:900">${_ai_pnl:+,.0f}</div></div>'
                    f'<div style="display:flex;gap:.8rem;margin-top:.4rem">'
                    f'<div><div style="color:#64748B;font-size:.63rem">WIN RATE</div>'
                    f'<div style="color:{"#10B981" if _ai_wr_p>=60 else "#F59E0B" if _ai_wr_p>=40 else "#EF4444"};font-size:.85rem;font-weight:800">{_ai_wr_p:.0f}%</div></div>'
                    f'<div><div style="color:#64748B;font-size:.63rem">TRADES</div>'
                    f'<div style="color:#F1F5F9;font-size:.85rem;font-weight:800">{_ai_n_tr}</div></div>'
                    f'<div><div style="color:#64748B;font-size:.63rem">AVG/TRADE</div>'
                    f'<div style="color:#F1F5F9;font-size:.85rem;font-weight:800">${_ai_avg:+,.0f}</div></div>'
                    f'</div></div>'
                    f'</div>'
                    # Winner banner
                    f'<div style="background:{"#0A1F0A" if _ai_better else "#0F1A0A" if not _ai_better else "#1A1400"};'
                    f'border-radius:6px;padding:.6rem .9rem;margin-top:.6rem;text-align:center">'
                    f'<span style="color:{_winner_color};font-size:.82rem;font-weight:900">'
                    f'{"🏆 AI RECOMMENDED STRATEGY WINS — consider switching parameters" if _ai_better else ("📌 YOUR STRATEGY IS FINE — your parameters perform as well or better" if not _ai_better else "🤝 TIE — both strategies performed equally")}'
                    f'</span>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # ── STRATEGY OPTIMIZER (button-based removed — now auto-runs in computation) ──
            # Display handled above, before dual backtest. No-op here.
            if False:
                _sopt_cache_key = (
                    f"sopt_{symbol}_{origin}_{target}_"
                    f"{opts_p.get('direction','')}_{opts_p.get('opt_type','')}"
                )
                st.markdown(
                    '<div style="background:linear-gradient(135deg,#0A0A05 0%,#14140A 100%);'
                    'border:2px solid #D97706;border-radius:12px;padding:.8rem 1.2rem;margin:.6rem 0 .4rem 0">'
                    '<div style="color:#F59E0B;font-size:.88rem;font-weight:900;letter-spacing:.05em">'
                    '🔍 STRATEGY OPTIMIZER — AI tests 12 real backtests to find historically best parameters</div>'
                    '<div style="color:#64748B;font-size:.72rem;margin-top:.2rem">'
                    'Runs 12 real TastyTrade backtests across different Delta · DTE · TP% · SL% combos '
                    'for the same symbol &amp; date range — ranked by P&amp;L. '
                    'Finds the best parameter set that would have worked historically for this exact window.</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )
                _sopt_btn_col, _sopt_inf_col = st.columns([1, 4])
                with _sopt_btn_col:
                    _sopt_trigger = st.button(
                        "🔍 Run Optimizer",
                        key=f"soptbtn_{abs(hash(_sopt_cache_key)) % 999999}",
                        help="Tests 12 parameter combos via real TT API (~90 sec). Results cached.",
                    )
                with _sopt_inf_col:
                    if st.session_state.get(_sopt_cache_key):
                        _sopt_cached_n = len(st.session_state[_sopt_cache_key])
                        st.caption(
                            f"✅ Cached results showing ({_sopt_cached_n} backtests). "
                            "Click 'Run Optimizer' to re-run with fresh data."
                        )
                    else:
                        st.caption(
                            "← Click to run 12 real TT backtests across different Delta/DTE/TP/SL "
                            "and find the parameter set that performed best for this stock and date range."
                        )

                if _sopt_trigger:
                    _sopt_dir_map  = {"Buy": "long", "Sell": "short", "buy": "long", "sell": "short"}
                    _sopt_type_map = {"Call": "call", "Put": "put", "call": "call", "put": "put"}
                    _sopt_dir      = _sopt_dir_map.get(opts_p.get("direction", "Sell"), "short")
                    _sopt_type     = _sopt_type_map.get(opts_p.get("opt_type", "Put"), "put")
                    _sopt_qty      = min(int(opts_p.get("quantity") or 1), 5)

                    _sopt_combos = [
                        (20,  7, 15,  5),
                        (20, 14, 20,  8),
                        (25,  7, 15,  5),
                        (25, 14, 25, 10),
                        (30, 14, 20,  8),
                        (30, 21, 30, 10),
                        (35, 14, 25, 10),
                        (35, 21, 30, 15),
                        (40, 14, 20, 10),
                        (40, 21, 35, 15),
                        (45, 21, 25, 10),
                        (50, 21, 40, 20),
                    ]

                    _sopt_rows   = []
                    _sopt_prog   = st.progress(0, text="Optimizer starting…")
                    _sopt_status = st.empty()

                    for _si, (_sd, _sdte, _stp, _ssl) in enumerate(_sopt_combos):
                        _sopt_status.info(
                            f"🔍 Combo {_si+1}/12: D{_sd} · {_sdte}DTE · TP{_stp}% · SL{_ssl}%"
                        )
                        _sopt_prog.progress(
                            int(_si / 12 * 100),
                            text=f"Testing combo {_si+1}/12…",
                        )
                        try:
                            _sopt_leg = {
                                "type": "equity-option",
                                "direction": _sopt_dir,
                                "quantity": _sopt_qty,
                                "side": _sopt_type,
                                "daysUntilExpiration": _sdte,
                                "strikeSelection": "delta",
                                "delta": _sd,
                            }
                            _sopt_payload = _tt_build_legs(
                                symbol=symbol,
                                start_date=origin,
                                end_date=target,
                                legs=[_sopt_leg],
                                entry_frequency="every day",
                                exit_rule="Exit at target date",
                                stop_loss_pct=float(_ssl),
                                take_profit_pct=float(_stp),
                            )
                            _sopt_id, _ = _tt_create_backtest(_sopt_payload)
                            if _sopt_id:
                                _sopt_resp, _ = _tt_poll_backtest(_sopt_id)
                                if _sopt_resp:
                                    _sr     = _sopt_resp.get("results") or {}
                                    _ss     = _sr.get("statistics") or {}
                                    _str    = _sr.get("trials") or []
                                    _sn     = int(_ss.get("Number of trades") or len(_str) or 0)
                                    _spnl   = float(_ss.get("Total profit/loss") or 0)
                                    _swrr   = float(_ss.get("Win percentage") or 0)
                                    _swr    = _swrr / 100.0 if _swrr > 1.0 else _swrr
                                    if _sn > 0:
                                        _sopt_rows.append({
                                            "delta": _sd, "dte": _sdte,
                                            "tp_pct": _stp, "sl_pct": _ssl,
                                            "qty": _sopt_qty,
                                            "total_pnl": _spnl,
                                            "win_rate": _swr,
                                            "trades": _sn,
                                            "avg_pnl": _spnl / _sn if _sn else 0,
                                        })
                        except Exception:
                            pass  # skip failed combos silently

                    _sopt_prog.progress(100, text="✅ Optimizer complete!")
                    _sopt_status.empty()

                    if _sopt_rows:
                        _sopt_rows.sort(key=lambda x: x["total_pnl"], reverse=True)
                        st.session_state[_sopt_cache_key] = _sopt_rows
                    else:
                        st.warning(
                            "Optimizer: TastyTrade API did not return results for any combination. "
                            "Check your credentials or try a different symbol."
                        )

                _sopt_display = st.session_state.get(_sopt_cache_key) or []
                if _sopt_display:
                    st.markdown(
                        '<div style="color:#F59E0B;font-size:.82rem;font-weight:800;margin:.7rem 0 .3rem 0">'
                        '📊 ALL 12 BACKTESTS — RANKED BEST → WORST</div>',
                        unsafe_allow_html=True,
                    )
                    _sopt_tbl = (
                        '<div style="overflow-x:auto;border-radius:8px;border:1px solid #1E3A5F;margin-bottom:.8rem">'
                        '<table style="width:100%;border-collapse:collapse;font-size:.74rem">'
                        '<thead><tr style="background:#0D1A2E;color:#94A3B8;font-weight:700">'
                        '<th style="padding:.5rem .6rem;text-align:center">RANK</th>'
                        '<th style="padding:.5rem .6rem;text-align:center">DELTA</th>'
                        '<th style="padding:.5rem .6rem;text-align:center">DTE</th>'
                        '<th style="padding:.5rem .6rem;text-align:center;color:#10B981">TP%</th>'
                        '<th style="padding:.5rem .6rem;text-align:center;color:#EF4444">SL%</th>'
                        '<th style="padding:.5rem .6rem;text-align:right">TOTAL P&L</th>'
                        '<th style="padding:.5rem .6rem;text-align:right">WIN RATE</th>'
                        '<th style="padding:.5rem .6rem;text-align:right">TRADES</th>'
                        '<th style="padding:.5rem .6rem;text-align:right">AVG/TRADE</th>'
                        '</tr></thead><tbody>'
                    )
                    for _sri, _srow in enumerate(_sopt_display):
                        _s_best  = _sri == 0
                        _s_worst = _sri == len(_sopt_display) - 1
                        _s_bg    = "#0A1F0A" if _s_best else ("#200A0A" if _s_worst else "#0A0F1E")
                        _s_bdr   = (
                            "border-left:3px solid #10B981;" if _s_best
                            else "border-left:3px solid #EF4444;" if _s_worst
                            else "border-left:3px solid #1E3A5F;"
                        )
                        _s_pclr  = "#10B981" if _srow["total_pnl"] >= 0 else "#EF4444"
                        _s_wrp   = _srow["win_rate"] * 100
                        _s_wclr  = "#10B981" if _s_wrp >= 60 else ("#F59E0B" if _s_wrp >= 40 else "#EF4444")
                        _s_aclr  = "#10B981" if _srow["avg_pnl"] >= 0 else "#EF4444"
                        _s_rank  = "🥇 #1 BEST" if _s_best else (f"💀 #{_sri+1} WORST" if _s_worst else f"#{_sri+1}")
                        _sopt_tbl += (
                            f'<tr style="background:{_s_bg};{_s_bdr}border-bottom:1px solid #1E3A5F">'
                            f'<td style="padding:.42rem .6rem;text-align:center;'
                            f'color:{"#F59E0B" if _s_best else "#94A3B8"};'
                            f'font-weight:{"900" if _s_best else "400"}">{_s_rank}</td>'
                            f'<td style="padding:.42rem .6rem;text-align:center;color:#F1F5F9;font-weight:700">{_srow["delta"]}</td>'
                            f'<td style="padding:.42rem .6rem;text-align:center;color:#F1F5F9">{_srow["dte"]}d</td>'
                            f'<td style="padding:.42rem .6rem;text-align:center;color:#10B981;font-weight:700">{_srow["tp_pct"]}%</td>'
                            f'<td style="padding:.42rem .6rem;text-align:center;color:#F59E0B;font-weight:700">{_srow["sl_pct"]}%</td>'
                            f'<td style="padding:.42rem .6rem;text-align:right;color:{_s_pclr};font-weight:800">${_srow["total_pnl"]:+,.0f}</td>'
                            f'<td style="padding:.42rem .6rem;text-align:right;color:{_s_wclr};font-weight:700">{_s_wrp:.0f}%</td>'
                            f'<td style="padding:.42rem .6rem;text-align:right;color:#CBD5E1">{_srow["trades"]}</td>'
                            f'<td style="padding:.42rem .6rem;text-align:right;color:{_s_aclr}">${_srow["avg_pnl"]:+,.0f}</td>'
                            f'</tr>'
                        )
                    _sopt_tbl += '</tbody></table></div>'
                    st.markdown(_sopt_tbl, unsafe_allow_html=True)

                    # Best vs User comparison
                    _sob       = _sopt_display[0]
                    _sob_pnl   = _sob["total_pnl"]
                    _sob_wrp   = _sob["win_rate"] * 100
                    _sob_n     = _sob["trades"]
                    _sob_avg   = _sob["avg_pnl"]
                    _sob_pclr  = "#10B981" if _sob_pnl >= 0 else "#EF4444"
                    _sob_wclr  = "#10B981" if _sob_wrp >= 60 else ("#F59E0B" if _sob_wrp >= 40 else "#EF4444")
                    _sob_diff  = _sob_pnl - _bt_pnl
                    _sob_wins  = _sob_pnl > _bt_pnl
                    _sob_tie   = abs(_sob_diff) < 1
                    _sob_lbl   = "OPTIMIZER WINS" if _sob_wins else ("TIE" if _sob_tie else "YOUR STRATEGY WINS")
                    _sob_clr   = "#F59E0B" if _sob_wins else ("#94A3B8" if _sob_tie else "#10B981")
                    _sob_utp   = float(opts_p.get("take_profit_pct") or 0)
                    _sob_usl   = float(opts_p.get("stop_loss_pct") or 0)
                    _sob_banner = (
                        f"🥇 USE OPTIMIZER PARAMS — D{_sob['delta']} · {_sob['dte']}DTE · "
                        f"TP{_sob['tp_pct']}% · SL{_sob['sl_pct']}% performed better historically"
                        if _sob_wins else
                        "📌 YOUR STRATEGY IS NEAR-OPTIMAL — optimizer found no significantly better combo for this period"
                    )

                    st.markdown(
                        f'<div style="background:linear-gradient(135deg,#0A0F1E 0%,#0D1A2E 100%);'
                        f'border:2px solid #D97706;border-radius:12px;padding:1rem 1.2rem;margin:.3rem 0 .6rem 0">'
                        f'<div style="color:#F59E0B;font-size:.88rem;font-weight:900;letter-spacing:.05em;'
                        f'margin-bottom:.4rem;border-bottom:1px solid #2A2010;padding-bottom:.5rem">'
                        f'🏆 YOUR STRATEGY vs OPTIMIZER BEST — FINAL COMPARISON</div>'
                        f'<div style="display:grid;grid-template-columns:1fr auto 1fr;gap:.8rem;align-items:center">'
                        f'<div style="background:#111827;border-radius:8px;padding:.8rem;border-top:3px solid #4B5563">'
                        f'<div style="color:#94A3B8;font-size:.68rem;font-weight:700;margin-bottom:.3rem">YOUR STRATEGY</div>'
                        f'<div style="color:#F1F5F9;font-size:.75rem;font-weight:700">'
                        f'{opts_p.get("direction","—")} {opts_p.get("opt_type","—")} · '
                        f'D{opts_p.get("delta","—")} · {opts_p.get("dte","—")}DTE · {opts_p.get("quantity","—")}qty</div>'
                        f'<div style="color:#94A3B8;font-size:.63rem;margin-bottom:.4rem">TP {_sob_utp:.0f}% · SL {_sob_usl:.0f}%</div>'
                        f'<div style="color:#64748B;font-size:.63rem">TOTAL P&L</div>'
                        f'<div style="color:{_bt_pnl_clr};font-size:1.2rem;font-weight:900">${_bt_pnl:+,.0f}</div>'
                        f'<div style="display:flex;gap:.8rem;margin-top:.4rem">'
                        f'<div><div style="color:#64748B;font-size:.63rem">WIN RATE</div>'
                        f'<div style="color:{_bt_wr_clr};font-size:.85rem;font-weight:800">{_bt_wr_pct:.0f}%</div></div>'
                        f'<div><div style="color:#64748B;font-size:.63rem">TRADES</div>'
                        f'<div style="color:#F1F5F9;font-size:.85rem;font-weight:800">{_bt_total_trades}</div></div>'
                        f'<div><div style="color:#64748B;font-size:.63rem">AVG/TRADE</div>'
                        f'<div style="color:#F1F5F9;font-size:.85rem;font-weight:800">'
                        f'${(_bt_pnl / _bt_total_trades if _bt_total_trades else 0):+,.0f}</div></div>'
                        f'</div></div>'
                        f'<div style="text-align:center">'
                        f'<div style="color:{_sob_clr};font-size:1.5rem;font-weight:900">VS</div>'
                        f'<div style="color:{_sob_clr};font-size:.65rem;font-weight:800;margin-top:.2rem">{_sob_lbl}</div>'
                        f'<div style="color:#475569;font-size:.62rem;margin-top:.15rem">diff: ${_sob_diff:+,.0f}</div>'
                        f'</div>'
                        f'<div style="background:#111827;border-radius:8px;padding:.8rem;border-top:3px solid #D97706">'
                        f'<div style="color:#F59E0B;font-size:.68rem;font-weight:700;margin-bottom:.3rem">🥇 OPTIMIZER BEST</div>'
                        f'<div style="color:#F1F5F9;font-size:.75rem;font-weight:700">'
                        f'{opts_p.get("direction","—")} {opts_p.get("opt_type","—")} · '
                        f'D{_sob["delta"]} · {_sob["dte"]}DTE · {_sob["qty"]}qty</div>'
                        f'<div style="color:#94A3B8;font-size:.63rem;margin-bottom:.4rem">TP {_sob["tp_pct"]}% · SL {_sob["sl_pct"]}%</div>'
                        f'<div style="color:#64748B;font-size:.63rem">TOTAL P&L</div>'
                        f'<div style="color:{_sob_pclr};font-size:1.2rem;font-weight:900">${_sob_pnl:+,.0f}</div>'
                        f'<div style="display:flex;gap:.8rem;margin-top:.4rem">'
                        f'<div><div style="color:#64748B;font-size:.63rem">WIN RATE</div>'
                        f'<div style="color:{_sob_wclr};font-size:.85rem;font-weight:800">{_sob_wrp:.0f}%</div></div>'
                        f'<div><div style="color:#64748B;font-size:.63rem">TRADES</div>'
                        f'<div style="color:#F1F5F9;font-size:.85rem;font-weight:800">{_sob_n}</div></div>'
                        f'<div><div style="color:#64748B;font-size:.63rem">AVG/TRADE</div>'
                        f'<div style="color:#F1F5F9;font-size:.85rem;font-weight:800">${_sob_avg:+,.0f}</div></div>'
                        f'</div></div>'
                        f'</div>'
                        f'<div style="background:{"#0A1F0A" if _sob_wins else "#0A1430"};'
                        f'border-radius:6px;padding:.55rem .9rem;margin-top:.6rem;text-align:center">'
                        f'<span style="color:{_sob_clr};font-size:.82rem;font-weight:900">{_sob_banner}</span>'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )

            # ── BIG FINAL VERDICT BANNER ──────────────────────────────────────
            _fv_bg = (
                "linear-gradient(135deg,#0A2010 0%,#0D2C14 100%)" if _final_verdict == "TRADE"
                else "linear-gradient(135deg,#1A1000 0%,#2C1A00 100%)" if _final_verdict == "MODIFY"
                else "linear-gradient(135deg,#1A0505 0%,#2C0A0A 100%)" if _final_verdict == "NO TRADE"
                else "linear-gradient(135deg,#0D0D20 0%,#1A1A2C 100%)"
            )
            _fv_border = (
                "#10B981" if _final_verdict == "TRADE"
                else "#F59E0B" if _final_verdict == "MODIFY"
                else "#EF4444" if _final_verdict == "NO TRADE"
                else "#6366F1"
            )
            _fv_sub = (
                f"AI decision = {_ai_dec_early} | Strategy = {_atm_strategy} | Alignment = {_rtc_align}"
            )
            st.markdown(
                f'<div style="background:{_fv_bg};border:3px solid {_fv_border};'
                f'border-radius:12px;padding:1rem 1.5rem;margin:.3rem 0 .8rem 0;text-align:center">'
                f'<div style="color:#64748B;font-size:.72rem;font-weight:700;letter-spacing:.1em;margin-bottom:.3rem">FINAL VERDICT</div>'
                f'<div style="color:{_final_verdict_color};font-size:2rem;font-weight:900;letter-spacing:.08em">{_final_verdict}</div>'
                f'<div style="color:#CBD5E1;font-size:.78rem;margin-top:.3rem">{_final_verdict_desc}</div>'
                f'<div style="color:#475569;font-size:.65rem;margin-top:.4rem">{_fv_sub}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div style="background:linear-gradient(135deg,#060D18 0%,#0A1830 100%);'
                f'border:2px solid #1E4D7A;border-radius:12px;padding:1.1rem 1.3rem;margin:.3rem 0 1rem 0">'
                # Header
                f'<div style="color:#60A5FA;font-size:.95rem;font-weight:900;letter-spacing:.05em;'
                f'margin-bottom:.8rem;border-bottom:1px solid #1E3A5F;padding-bottom:.5rem">'
                f'🎯 AGENT SELECTS CONFIGURATION → BACKTESTER VALIDATES → COMPARISON</div>'
                # Step 1: Agent
                f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:.8rem;margin-bottom:.8rem">'
                f'<div style="background:#0A1F3A;border-radius:8px;padding:.7rem .9rem">'
                f'<div style="color:#94A3B8;font-size:.68rem;font-weight:700;letter-spacing:.05em;margin-bottom:.4rem">STEP 1 — AI AGENT SELECTED</div>'
                f'<div style="color:#FCD34D;font-size:.85rem;font-weight:900">{_atm_strategy.upper()}</div>'
                f'<div style="color:#94A3B8;font-size:.72rem;margin-top:.3rem">Symbol: <b style="color:#F1F5F9">{symbol}</b></div>'
                f'<div style="color:#94A3B8;font-size:.72rem">Delta: <b style="color:#F1F5F9">{opts_p.get("delta","—")}</b> &nbsp;|&nbsp; '
                f'DTE: <b style="color:#F1F5F9">{opts_p.get("dte","—")}d</b> &nbsp;|&nbsp; '
                f'Qty: <b style="color:#F1F5F9">{opts_p.get("quantity","—")}</b></div>'
                f'<div style="color:#94A3B8;font-size:.72rem;margin-top:.2rem">'
                f'AI: <b style="color:{("#10B981" if _dec_upper=="BUY" else "#EF4444" if _dec_upper=="SELL" else "#F59E0B")}">'
                f'{_dec_upper}</b> · Return: <b style="color:{("#10B981" if _atm_ai_ret>=0 else "#EF4444")}">'
                f'{_atm_ai_ret:+.1f}%</b> · Conf: <b style="color:#FCD34D">{_atm_ai_conf}/100</b></div>'
                f'</div>'
                # Step 2: Backtester
                f'<div style="background:#0A1F3A;border-radius:8px;padding:.7rem .9rem">'
                f'<div style="color:#94A3B8;font-size:.68rem;font-weight:700;letter-spacing:.05em;margin-bottom:.4rem">STEP 2 — BACKTESTER EXECUTED</div>'
                f'<div style="color:#60A5FA;font-size:.85rem;font-weight:900">{_bt_total_trades} TRADES</div>'
                f'<div style="display:flex;gap:.8rem;margin-top:.4rem;flex-wrap:wrap">'
                f'<div><div style="color:#64748B;font-size:.65rem">TOTAL P&L</div>'
                f'<div style="color:{_bt_pnl_clr};font-size:.9rem;font-weight:800">${_bt_pnl:+,.0f}</div></div>'
                f'<div><div style="color:#64748B;font-size:.65rem">WIN RATE</div>'
                f'<div style="color:{_bt_wr_clr};font-size:.9rem;font-weight:800">{_bt_wr_pct:.0f}%</div></div>'
                f'<div><div style="color:#64748B;font-size:.65rem">AVG/TRADE</div>'
                f'<div style="color:#F1F5F9;font-size:.9rem;font-weight:800">${(_bt_pnl/_bt_total_trades if _bt_total_trades else 0):+,.0f}</div></div>'
                f'</div></div>'
                # Step 3: Verdict
                f'<div style="background:#0A1F3A;border-radius:8px;padding:.7rem .9rem">'
                f'<div style="color:#94A3B8;font-size:.68rem;font-weight:700;letter-spacing:.05em;margin-bottom:.4rem">STEP 3 — FINAL VERDICT</div>'
                f'<div style="color:{_final_verdict_color};font-size:1.05rem;font-weight:900;letter-spacing:.04em">{_final_verdict_icon}</div>'
                f'<div style="color:#CBD5E1;font-size:.68rem;margin-top:.2rem;line-height:1.4">{_final_verdict_desc}</div>'
                f'<div style="margin-top:.4rem">'
                f'<span style="background:{_align_badge_color}22;color:{_align_badge_color};'
                f'border:1px solid {_align_badge_color};border-radius:4px;padding:.1rem .4rem;font-size:.68rem;font-weight:700">'
                f'AI vs USER: {_rtc_align}</span></div>'
                f'{_rtc_notes_html}'
                f'</div>'
                f'</div>'
                # AI Recommended config row
                f'<div style="background:#111827;border-radius:8px;padding:.6rem .9rem;border-left:3px solid #6366F1">'
                f'<span style="color:#94A3B8;font-size:.7rem;font-weight:700">AI AGENT RECOMMENDS: </span>'
                f'<span style="color:#A78BFA;font-weight:800;font-size:.85rem">{_rtc_action}</span>'
                f' &nbsp;·&nbsp; <span style="color:#64748B">Symbol</span> <span style="color:#F1F5F9;font-weight:700">{symbol}</span>'
                f' &nbsp;·&nbsp; <span style="color:#64748B">Delta</span> <span style="color:#F1F5F9;font-weight:700">{_rtc_delta}</span>'
                f' &nbsp;·&nbsp; <span style="color:#64748B">DTE</span> <span style="color:#F1F5F9;font-weight:700">{_rtc_dte}d</span>'
                f' &nbsp;·&nbsp; <span style="color:#64748B">Qty</span> <span style="color:#F1F5F9;font-weight:700">{_rtc_qty} contracts</span>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # ── API inflation warning for short strategies ─────────────────────
            # TastyTrade API returns inflated win rates for Sell Put/Call with tight SL.
            # Example: SL=5%, real win rate = 44% (from website), API returns 94%.
            # Warn the user to verify inflated results on the TastyTrade website.
            if (
                _is_sell_strategy
                and not opts.get("bs_fallback")
                and _bt_wr_pct > 80
                and float(opts_p.get("stop_loss_pct") or 100) < 25
            ):
                _sl_pct_warn = float(opts_p.get("stop_loss_pct") or 0)
                st.warning(
                    f"⚠️ **API RESULT MAY BE INFLATED** — TastyTrade API does not correctly apply "
                    f"tight stop-loss ({_sl_pct_warn:.0f}% SL) for short option strategies. "
                    f"With SL={_sl_pct_warn:.0f}% the realistic win rate is **40–65%**, not {_bt_wr_pct:.0f}%. "
                    f"**Verify this backtest on the TastyTrade website** — the website applies SL/TP rules "
                    f"precisely and will show the correct (lower) P&L."
                )

            # ── Agent entry logic — OPTIONS STRATEGY AWARE ───────────────────────
            # Use the actual AI decision (BUY/SELL/HOLD/REVIEW), NOT raw return comparison.
            # Correct options logic:
            #   Buy Call   → needs BUY (bullish conviction)
            #   Buy Put    → needs SELL (bearish conviction)
            #   Sell Put   → needs BUY or HOLD (neutral/bullish is fine — just can't crash)
            #   Sell Call  → needs SELL or HOLD (neutral/bearish is fine — just can't surge)
            _ai_dec_str       = str(_atm_ai_dec).upper()
            _ai_is_buy_dec    = _ai_dec_str == "BUY"
            _ai_is_sell_dec   = _ai_dec_str == "SELL"
            _ai_is_hold_atm   = _ai_dec_str == "HOLD"
            _ai_is_review_atm = _ai_dec_str == "REVIEW"
            # Legacy aliases (used elsewhere)
            _ai_is_bullish_atm = _ai_is_buy_dec
            _ai_is_bearish_atm = _ai_is_sell_dec

            # _is_sell_strategy and _is_buy_strategy are already set above

            if _atm_dir == "buy" and _atm_type == "call":
                _agent_enters = _ai_is_buy_dec and not _ai_is_review_atm
                _agent_logic  = (
                    "Agent enters BUY CALL only when AI decision = BUY "
                    "(needs strong bullish conviction — stock must rise above strike)"
                )
            elif _atm_dir == "buy" and _atm_type == "put":
                _agent_enters = _ai_is_sell_dec and not _ai_is_review_atm
                _agent_logic  = (
                    "Agent enters BUY PUT only when AI decision = SELL "
                    "(needs strong bearish conviction — stock must fall below strike)"
                )
            elif _atm_dir == "sell" and _atm_type == "put":
                # SELL PUT profits when stock is neutral or bullish (premium decays to zero)
                # HOLD = neutral = fine for Sell Put. Only skip on SELL (strong bearish) or REVIEW.
                _agent_enters = not _ai_is_sell_dec and not _ai_is_review_atm
                _agent_logic  = (
                    "Agent enters SELL PUT when AI = BUY or HOLD "
                    "(neutral/bullish OK — stock just needs to NOT crash below the put strike)"
                )
            elif _atm_dir == "sell" and _atm_type == "call":
                # SELL CALL profits when stock is neutral or bearish
                _agent_enters = not _ai_is_buy_dec and not _ai_is_review_atm
                _agent_logic  = (
                    "Agent enters SELL CALL when AI = SELL or HOLD "
                    "(neutral/bearish OK — stock just needs to NOT surge above the call strike)"
                )
            else:
                _agent_enters = not _ai_is_review_atm
                _agent_logic  = "Agent enters when AI decision is not REVIEW"

            # Action label — for premium sellers, HOLD is a valid entry signal
            if _agent_enters:
                _agent_action_label = (
                    "ENTER (HOLD=VALID)" if _ai_is_hold_atm and _is_sell_strategy
                    else "ENTER"
                )
            elif _ai_is_review_atm:
                _agent_action_label = "⚠️ REVIEW — low confidence"
            elif _is_sell_strategy and _ai_is_sell_dec:
                _agent_action_label = "⚠️ AI BEARISH — Sell Put has crash risk"
            elif _is_sell_strategy and _ai_is_buy_dec and _atm_type == "call":
                _agent_action_label = "⚠️ AI BULLISH — Sell Call has surge risk"
            elif _ai_is_hold_atm:
                _agent_action_label = "⚠️ AI HOLD — no strong directional conviction"
            else:
                _agent_action_label = "⚠️ AI CONFLICTS with strategy"

            _agent_action_color = (
                "#10B981" if _agent_enters
                else "#F59E0B" if _ai_is_hold_atm
                else "#EF4444"
            )

            # Banner
            st.markdown(
                f'<div style="background:#060D18;border:2px solid #1E3A5F;border-radius:10px;'
                f'padding:.9rem 1.2rem;margin:.4rem 0 .8rem 0">'
                f'<div style="color:#60A5FA;font-size:.9rem;font-weight:900;letter-spacing:.06em;margin-bottom:.6rem">'
                f'AGENT TRADE MAP — HOW MANY TRADES WOULD THE AGENT TAKE?</div>'
                f'<div style="display:flex;gap:1.5rem;flex-wrap:wrap;font-size:.8rem">'
                f'<div><span style="color:#64748B">AI Decision:</span> '
                f'<span style="font-weight:900">{_decision_badge(_atm_ai_dec)}</span></div>'
                f'<div><span style="color:#64748B">Predicted Return:</span> '
                f'<span style="color:{"#10B981" if _atm_ai_ret>=0 else "#EF4444"};font-weight:700">'
                f'{_sign_fmt(_atm_ai_ret, suffix="%")}</span></div>'
                f'<div><span style="color:#64748B">Confidence:</span> '
                f'<span style="color:#FCD34D;font-weight:700">{_atm_ai_conf}/100</span></div>'
                f'<div><span style="color:#64748B">Strategy:</span> '
                f'<span style="color:#6EE7B7;font-weight:700">{_atm_strategy}</span></div>'
                f'</div>'
                f'<div style="margin-top:.5rem;font-size:.78rem;color:#94A3B8">'
                f'Logic: {_agent_logic}</div>'
                f'<div style="margin-top:.4rem;font-size:.82rem;font-weight:700">'
                f'Agent Action: <span style="color:{_agent_action_color};font-size:.95rem;'
                f'font-weight:900">{_agent_action_label}</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # ── HOLD explanation box — only show for premium BUYERS (Buy Call/Put) ────
            # For premium SELLERS (Sell Put/Call), HOLD = ENTER so no warning needed.
            if _ai_is_hold_atm and _is_buy_strategy:
                st.markdown(
                    f'<div style="background:#1C1009;border:2px solid #F59E0B;border-radius:8px;'
                    f'padding:.9rem 1.2rem;margin:.4rem 0 .6rem 0">'
                    f'<div style="color:#FCD34D;font-weight:900;font-size:.9rem;margin-bottom:.4rem">'
                    f'HOLD — PREMIUM BUYER SKIPS THIS TRADE</div>'
                    f'<div style="color:#FED7AA;font-size:.8rem;line-height:1.6">'
                    f'AI returned <b>HOLD</b> — no strong directional conviction for the {horizon}-day window.<br>'
                    f'For a <b>{_atm_strategy}</b> position (premium buyer), the agent needs a strong directional '
                    f'signal (BUY for Call, SELL for Put) — time decay works against you when the market is flat.<br>'
                    f'<span style="color:#FCD34D">Agent would skip all {len(_raw_trials) if _raw_trials else "N"} backtester trades.</span>'
                    f'</div>'
                    f'<div style="margin-top:.6rem;font-size:.75rem;color:#94A3B8">'
                    f'Tip: For HOLD signals, consider switching to a Sell Put/Call strategy — premium sellers profit from neutral markets.</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            elif _ai_is_hold_atm and _is_sell_strategy:
                st.markdown(
                    f'<div style="background:#0A2010;border:2px solid #10B981;border-radius:8px;'
                    f'padding:.9rem 1.2rem;margin:.4rem 0 .6rem 0">'
                    f'<div style="color:#6EE7B7;font-weight:900;font-size:.9rem;margin-bottom:.4rem">'
                    f'HOLD = VALID ENTRY FOR PREMIUM SELLER ✅</div>'
                    f'<div style="color:#A7F3D0;font-size:.8rem;line-height:1.6">'
                    f'AI returned <b>HOLD</b> — neutral market expected. This is <b>ideal</b> for a <b>{_atm_strategy}</b>.<br>'
                    f'Premium sellers profit when the market stays flat or moves modestly — time decay works FOR you.<br>'
                    f'The agent <b>ENTERS</b> this trade because HOLD = "no strong directional move expected" = perfect for selling premium.'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            if not _raw_trials:
                st.info("No individual trade data returned by TastyTrade backtester for this run.")
            else:
                # Build per-trade AI accuracy table
                # AI makes ONE signal for the period → show how accurate it was per trade
                _atm_rows    = []
                _n_ai_correct = 0
                _total_pl_all = 0.0
                _total_wins   = 0

                # AI signal label for display
                _ai_sig_lbl = "BUY"  if _ai_is_buy_dec  else ("SELL" if _ai_is_sell_dec else "HOLD")
                _ai_sig_clr = "#10B981" if _ai_is_buy_dec else ("#EF4444" if _ai_is_sell_dec else "#F59E0B")

                for _i, _t in enumerate(_raw_trials, 1):
                    _topen  = str(_t.get("openDateTime") or _t.get("entryDate") or _t.get("entry_date") or "")[:10]
                    _tclose = str(_t.get("closeDateTime") or _t.get("exitDate") or _t.get("exit_date") or "")[:10]
                    _tpl    = float(_t.get("profitLoss") or _t.get("profit_loss") or 0)
                    _troi   = float(_t.get("returnOnInvestment") or _t.get("roi") or 0)
                    _treason = str(_t.get("exitReason") or _t.get("closeReason") or _t.get("exit_reason") or "---").replace("_", " ").title()
                    try:
                        from datetime import date as _dt_cls
                        _days_held = (_dt_cls.fromisoformat(_tclose) - _dt_cls.fromisoformat(_topen)).days if (_topen and _tclose) else "?"
                    except Exception:
                        _days_held = "?"

                    _trade_won = _tpl > 0
                    _total_wins   += 1 if _trade_won else 0
                    _total_pl_all += _tpl

                    # AI correct = (agent would enter AND trade won) OR (agent would NOT enter AND trade lost)
                    # In both cases the AI signal matched what actually happened
                    _ai_correct = (_agent_enters == _trade_won)
                    if _ai_correct:
                        _n_ai_correct += 1

                    _res_lbl   = "WIN ✅" if _trade_won else "LOSS ❌"
                    _align_lbl = "✅ Correct" if _ai_correct else "❌ Wrong"

                    _atm_rows.append({
                        "#":           _i,
                        "Entry Date":  _topen or "---",
                        "Exit Date":   _tclose or "---",
                        "Days":        _days_held,
                        "P&L":         f"${_tpl:+,.2f}",
                        "ROI":         f"{_troi:+.1f}%" if _troi else "---",
                        "Exit Reason": _treason,
                        "AI Signal":   _ai_sig_lbl,
                        "Result":      _res_lbl,
                        "AI Correct?": _align_lbl,
                    })

                # Score banner — unified AI accuracy view (no SKIP/ENTER, just accuracy)
                _n_total     = len(_raw_trials)
                _n_ai_wrong  = _n_total - _n_ai_correct
                _ai_acc_pct  = (_n_ai_correct / _n_total * 100) if _n_total else 0
                _ai_acc_clr  = "#10B981" if _ai_acc_pct >= 60 else ("#F59E0B" if _ai_acc_pct >= 40 else "#EF4444")
                _bt_wr_pct2  = (_total_wins / _n_total * 100) if _n_total else 0
                _bt_wr_clr2  = "#10B981" if _bt_wr_pct2 >= 60 else ("#F59E0B" if _bt_wr_pct2 >= 40 else "#EF4444")
                _pl_clr_all  = "#10B981" if _total_pl_all >= 0 else "#EF4444"
                st.markdown(
                    f'<div style="display:flex;gap:1rem;flex-wrap:wrap;margin:.4rem 0 .7rem 0">'
                    f'<div style="background:#0A2540;border-radius:8px;padding:.6rem 1rem;text-align:center;flex:1">'
                    f'<div style="color:#64748B;font-size:.7rem">AI SIGNAL (PERIOD)</div>'
                    f'<div style="color:{_ai_sig_clr};font-size:1.5rem;font-weight:900">{_ai_sig_lbl}</div>'
                    f'<div style="color:#64748B;font-size:.68rem">for entire {horizon}-day window</div></div>'
                    f'<div style="background:#0A2540;border-radius:8px;padding:.6rem 1rem;text-align:center;flex:1">'
                    f'<div style="color:#64748B;font-size:.7rem">AI ACCURACY</div>'
                    f'<div style="color:{_ai_acc_clr};font-size:1.5rem;font-weight:900">{_n_ai_correct}/{_n_total}</div>'
                    f'<div style="color:#64748B;font-size:.68rem">({_ai_acc_pct:.0f}%) — trades AI predicted correctly</div></div>'
                    f'<div style="background:#0A2540;border-radius:8px;padding:.6rem 1rem;text-align:center;flex:1">'
                    f'<div style="color:#64748B;font-size:.7rem">BACKTESTER WIN RATE</div>'
                    f'<div style="color:{_bt_wr_clr2};font-size:1.5rem;font-weight:900">{_bt_wr_pct2:.0f}%</div>'
                    f'<div style="color:#64748B;font-size:.68rem">{_total_wins} wins / {_n_total} trades</div></div>'
                    f'<div style="background:#0A2540;border-radius:8px;padding:.6rem 1rem;text-align:center;flex:1">'
                    f'<div style="color:#64748B;font-size:.7rem">BACKTESTER P&L</div>'
                    f'<div style="color:{_pl_clr_all};font-size:1.5rem;font-weight:900">${_total_pl_all:+,.0f}</div>'
                    f'<div style="color:#64748B;font-size:.68rem">all {_n_total} trades taken</div></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Per-trade table
                import pandas as pd
                import io as _io
                _atm_df = pd.DataFrame(_atm_rows)
                st.dataframe(
                    _atm_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "#":            st.column_config.NumberColumn("#", width="small"),
                        "P&L":          st.column_config.TextColumn("P&L"),
                        "AI Signal":    st.column_config.TextColumn("AI Signal"),
                        "Result":       st.column_config.TextColumn("Trade Result"),
                        "AI Correct?":  st.column_config.TextColumn("AI Correct?", width="medium"),
                    }
                )

                # Excel export — builds a formatted .xlsx in memory for download
                try:
                    _xls_buf = _io.BytesIO()
                    with pd.ExcelWriter(_xls_buf, engine="openpyxl") as _xw:
                        _atm_df.to_excel(_xw, sheet_name="Trades", index=False)
                        _ws = _xw.sheets["Trades"]
                        # Auto-fit column widths
                        for _col_cells in _ws.columns:
                            _max_w = max(len(str(_c.value or "")) for _c in _col_cells)
                            _ws.column_dimensions[_col_cells[0].column_letter].width = min(_max_w + 4, 30)
                        # Summary sheet
                        _sum_data = {
                            "Metric": ["Symbol", "Strategy", "Backtest Window", "Total Trades",
                                       "Wins", "Losses", "Win Rate", "AI Correct", "AI Accuracy",
                                       "Total P&L", "Avg P&L / Trade"],
                            "Value": [
                                str(symbol),
                                f"{_atm_dir.title()} {_atm_type.title()}",
                                f"{origin} → {target}",
                                _n_total,
                                _n_ai_correct if False else sum(1 for r in _atm_rows if "WIN" in str(r.get("Result",""))),
                                sum(1 for r in _atm_rows if "LOSS" in str(r.get("Result",""))),
                                f"{sum(1 for r in _atm_rows if 'WIN' in str(r.get('Result',''))) / _n_total * 100:.1f}%" if _n_total else "0%",
                                _n_ai_correct,
                                f"{_ai_acc_pct:.1f}%",
                                f"${float(opts.get('profit_loss') or 0):+,.2f}",
                                f"${float(opts.get('avg_pnl') or 0):+,.2f}",
                            ]
                        }
                        pd.DataFrame(_sum_data).to_excel(_xw, sheet_name="Summary", index=False)
                    _xls_buf.seek(0)
                    _xls_fname = f"{symbol}_{_atm_dir}_{_atm_type}_DTE{int(opts_p.get('dte',0))}_{origin[:10]}_{target[:10]}.xlsx"
                    st.download_button(
                        label="Download Trades Excel",
                        data=_xls_buf.getvalue(),
                        file_name=_xls_fname,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="dl_trades_excel",
                    )
                except Exception as _xls_e:
                    pass

                # ── AI SIGNAL VALIDATION TABLE ──────────────────────────────────────
                # Compare AI-directed BS trades against user trades, trade by trade.
                # Uses ai_comparison_trials (always computed via BS, no TT auth needed).
                _ai_sr = opts.get("ai_strategy_result") or {}
                # Primary: BS comparison (computed always). Fallback: second TT backtest trials.
                _ai_cmp_trials = opts.get("ai_comparison_trials") or _ai_sr.get("trials") or []
                _ai_cmp_action = str(opts.get("ai_comparison_action") or _ai_sr.get("action") or "—")
                _ai_cmp_pnl    = float(opts.get("ai_comparison_profit_loss") or _ai_sr.get("profit_loss") or 0)
                _ai_cmp_wr     = float(opts.get("ai_comparison_win_rate") or _ai_sr.get("win_rate") or 0)
                _ai_cmp_delta  = opts.get("ai_comparison_delta") or _ai_sr.get("delta", "—")
                _ai_cmp_dte    = opts.get("ai_comparison_dte") or _ai_sr.get("dte", "—")
                _ai_cmp_dte_orig = opts.get("ai_comparison_dte_original")
                _ai_cmp_err    = opts.get("ai_comparison_error") or ""
                _ai_sr_trials  = _ai_cmp_trials  # alias used by download block below
                if _ai_cmp_err and not _ai_cmp_trials:
                    st.caption(f"AI comparison unavailable: {_ai_cmp_err}")
                # ── Strategy + AI Signal Reference Card ─────────────────────────
                st.markdown(
                    '<div style="background:#0A1428;border:1px solid #1E3A5F;border-radius:10px;'
                    'padding:.8rem 1.1rem;margin:.3rem 0 .7rem 0">'
                    '<div style="color:#60A5FA;font-size:.82rem;font-weight:900;margin-bottom:.5rem;'
                    'letter-spacing:.04em">📋 STRATEGY + AI SIGNAL — WHAT EACH COMBINATION MEANS</div>'
                    '<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:.4rem .8rem;font-size:.70rem">'

                    '<div style="color:#10B981;font-weight:700">✅ BUY CALL + AI=BUY → AGENT ENTERS</div>'
                    '<div style="color:#94A3B8">Both bullish. Stock must rise above call strike to profit.</div>'

                    '<div style="color:#EF4444;font-weight:700">❌ BUY CALL + AI=SELL → CONFLICT, NO ENTRY</div>'
                    '<div style="color:#94A3B8">AI says market falls — BUY CALL loses when market falls. Skip.</div>'

                    '<div style="color:#EF4444;font-weight:700">❌ BUY CALL + AI=HOLD → CONFLICT, NO ENTRY</div>'
                    '<div style="color:#94A3B8">BUY CALL needs strong bullish conviction. HOLD is not enough.</div>'

                    '<div style="color:#10B981;font-weight:700">✅ SELL PUT + AI=BUY → AGENT ENTERS</div>'
                    '<div style="color:#94A3B8">Both bullish. Sell Put profits when stock stays above put strike.</div>'

                    '<div style="color:#10B981;font-weight:700">✅ SELL PUT + AI=HOLD → AGENT ENTERS</div>'
                    '<div style="color:#94A3B8">Neutral is fine for Sell Put. Stock just needs to NOT crash below strike.</div>'

                    '<div style="color:#EF4444;font-weight:700">❌ SELL PUT + AI=SELL → CONFLICT, NO ENTRY</div>'
                    '<div style="color:#94A3B8">AI says crash risk — Sell Put loses if stock falls hard. Skip.</div>'

                    '<div style="color:#10B981;font-weight:700">✅ BUY PUT + AI=SELL → AGENT ENTERS</div>'
                    '<div style="color:#94A3B8">Both bearish. Stock must fall below put strike to profit.</div>'

                    '<div style="color:#EF4444;font-weight:700">❌ BUY PUT + AI=BUY → CONFLICT, NO ENTRY</div>'
                    '<div style="color:#94A3B8">AI says market rises — BUY PUT loses when market rises. Skip.</div>'

                    '<div style="color:#10B981;font-weight:700">✅ SELL CALL + AI=SELL → AGENT ENTERS</div>'
                    '<div style="color:#94A3B8">Both bearish/neutral. Sell Call profits when stock stays below call strike.</div>'

                    '<div style="color:#10B981;font-weight:700">✅ SELL CALL + AI=HOLD → AGENT ENTERS</div>'
                    '<div style="color:#94A3B8">Neutral is fine for Sell Call. Stock just needs to NOT surge above strike.</div>'

                    '</div>'
                    '<div style="color:#F59E0B;font-size:.68rem;margin-top:.5rem;border-top:1px solid #1E3A5F;padding-top:.4rem">'
                    '⚠️ NOTE: "AI Trade: Sell Put" in the comparison table does NOT mean AI signals SELL on the stock. '
                    '"Sell Put" is a neutral/bullish options strategy — it profits when stock stays flat or rises. '
                    'Both "Buy Call" and "Sell Put" are bullish. The AI-directed trade is the strategy that best fits the AI signal.</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )

                if _ai_cmp_trials and _raw_trials:
                    st.markdown(
                        '<div style="background:linear-gradient(135deg,#0A0F1E 0%,#0D1A2E 100%);'
                        'border:2px solid #7C3AED;border-radius:12px;padding:1rem 1.3rem;margin:.6rem 0 .5rem 0">'
                        '<div style="color:#A78BFA;font-size:.92rem;font-weight:900;letter-spacing:.05em;'
                        'margin-bottom:.3rem">🔬 AI SIGNAL VALIDATION — TRADE-BY-TRADE COMPARISON</div>'
                        '<div style="color:#64748B;font-size:.73rem">'
                        f'AI said <b style="color:{_ai_sig_clr}">{_ai_sig_lbl}</b> → Black-Scholes ran '
                        f'<b style="color:#A78BFA">{_ai_cmp_action.title()}</b> (D{_ai_cmp_delta} DTE{_ai_cmp_dte}'
                        + (f', capped from AI-suggested DTE{_ai_cmp_dte_orig}' if _ai_cmp_dte_orig and int(_ai_cmp_dte_orig) != int(_ai_cmp_dte) else '')
                        + f'). Each row compares AI-directed trade P&amp;L against your strategy P&amp;L.</div>'
                        '<div style="color:#F59E0B;font-size:.70rem;margin-top:.35rem">'
                        '⚠️ "AI Trade P&L" = Black-Scholes theoretical simulation of what the AI-directed strategy would have earned per 5-day trade. '
                        '"User Trade P&L" = TastyTrade API backtest result. '
                        'Both are estimates — the TastyTrade API result may differ from TastyTrade website due to TP/SL engine differences.</div>'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                    # Build lookup: entry_date → AI-directed trial (primary: AI-suggested DTE)
                    _ai_trial_map: dict = {}
                    for _at in _ai_cmp_trials:
                        _at_date = str(_at.get("openDateTime") or _at.get("entryDate") or _at.get("entry_date") or "")[:10]
                        if _at_date:
                            _ai_trial_map[_at_date] = _at

                    # Secondary lookup: AI direction at user's DTE (fills dates beyond AI-DTE window)
                    _ai_cmp_trials_u = opts.get("ai_comparison_trials_user_dte") or []
                    _ai_trial_map_u: dict = {}
                    for _at in _ai_cmp_trials_u:
                        _at_date_u = str(_at.get("openDateTime") or _at.get("entryDate") or _at.get("entry_date") or "")[:10]
                        if _at_date_u and _at_date_u not in _ai_trial_map:
                            _ai_trial_map_u[_at_date_u] = _at
                    _ai_cmp_user_dte = opts.get("ai_comparison_user_dte")

                    _val_rows = []
                    _val_ai_correct = 0
                    _val_user_correct = 0

                    for _vt in _raw_trials:
                        _vt_open  = str(_vt.get("openDateTime") or _vt.get("entryDate") or _vt.get("entry_date") or "")[:10]
                        _vt_close = str(_vt.get("closeDateTime") or _vt.get("exitDate") or _vt.get("exit_date") or "")[:10]
                        _vt_pl    = float(_vt.get("profitLoss") or _vt.get("profit_loss") or 0)
                        _vt_rsn   = str(_vt.get("exitReason") or _vt.get("closeReason") or "---").replace("_", " ").title()

                        # Look up AI-directed trade for the same entry date (primary then supplemental)
                        _at_match  = _ai_trial_map.get(_vt_open)
                        _at_approx = False
                        if _at_match is None:
                            _at_match  = _ai_trial_map_u.get(_vt_open)
                            _at_approx = _at_match is not None
                        if _at_match is not None:
                            _at_pl    = float(_at_match.get("profitLoss") or _at_match.get("profit_loss") or 0)
                            _at_rsn   = str(_at_match.get("exitReason") or _at_match.get("closeReason") or "---").replace("_", " ").title()
                            _at_found = True
                        else:
                            _at_pl    = None
                            _at_rsn   = "—"
                            _at_found = False

                        _user_win = _vt_pl > 0
                        _ai_win   = (_at_pl is not None and _at_pl > 0)

                        # AI direction correct = AI-directed trade was profitable
                        _ai_dir_ok = _ai_win if _at_found else None
                        # Did AI and user agree on outcome?
                        _both_match = (_user_win == _ai_win) if _at_found else None

                        if _ai_dir_ok is True:
                            _val_ai_correct += 1
                        if _user_win:
                            _val_user_correct += 1

                        # "†" marks entries from the supplemental pass (user's DTE, AI direction)
                        _at_pl_str = (
                            f"${_at_pl:+,.2f}" + (" †" if _at_approx else "")
                            if _at_pl is not None else "—"
                        )
                        _val_rows.append({
                            "Entry Date":       _vt_open or "---",
                            "Exit Date":        _vt_close or "---",
                            "AI Signal":        _ai_sig_lbl,
                            "AI Trade":         _ai_cmp_action.title(),
                            "AI Trade P&L":     _at_pl_str,
                            "AI Trade Exit":    _at_rsn,
                            "AI Direction ✓":   ("✅ YES" if _ai_dir_ok else ("❌ NO" if _ai_dir_ok is False else "—")),
                            "User Trade":       f"{_atm_dir.title()} {_atm_type.title()}",
                            "User Trade P&L":   f"${_vt_pl:+,.2f}",
                            "User Trade Exit":  _vt_rsn,
                            "User Correct":     ("✅ WIN" if _user_win else "❌ LOSS"),
                            "Outcomes Match":   ("✅ MATCH" if _both_match else ("❌ DIFFER" if _both_match is False else "—")),
                        })

                    _val_n         = len(_val_rows)
                    _val_n_matched = sum(1 for r in _val_rows if r["AI Trade P&L"] != "—")
                    _val_ai_pct    = (_val_ai_correct / _val_n_matched * 100) if _val_n_matched else 0
                    _val_u_pct     = (_val_user_correct / _val_n * 100) if _val_n else 0
                    _val_ai_c      = "#10B981" if _val_ai_pct >= 60 else ("#F59E0B" if _val_ai_pct >= 40 else "#EF4444")
                    _val_u_c       = "#10B981" if _val_u_pct >= 60 else ("#F59E0B" if _val_u_pct >= 40 else "#EF4444")
                    _ai_cmp_pc     = "#10B981" if _ai_cmp_pnl >= 0 else "#EF4444"

                    st.markdown(
                        f'<div style="display:flex;gap:.8rem;flex-wrap:wrap;margin:.3rem 0 .6rem 0">'
                        f'<div style="background:#0A2540;border-radius:8px;padding:.55rem .9rem;text-align:center;flex:1">'
                        f'<div style="color:#64748B;font-size:.66rem">AI SIGNAL</div>'
                        f'<div style="color:{_ai_sig_clr};font-size:1.3rem;font-weight:900">{_ai_sig_lbl}</div>'
                        f'<div style="color:#64748B;font-size:.63rem">{_ai_cmp_action.title()}</div></div>'
                        f'<div style="background:#0A2540;border-radius:8px;padding:.55rem .9rem;text-align:center;flex:1">'
                        f'<div style="color:#64748B;font-size:.66rem">AI DIRECTION CORRECT</div>'
                        f'<div style="color:{_val_ai_c};font-size:1.3rem;font-weight:900">{_val_ai_correct}/{_val_n_matched}</div>'
                        f'<div style="color:#64748B;font-size:.63rem">({_val_ai_pct:.0f}%) — AI-directed trades profitable</div></div>'
                        f'<div style="background:#0A2540;border-radius:8px;padding:.55rem .9rem;text-align:center;flex:1">'
                        f'<div style="color:#64748B;font-size:.66rem">AI-DIRECTED P&L</div>'
                        f'<div style="color:{_ai_cmp_pc};font-size:1.3rem;font-weight:900">${_ai_cmp_pnl:+,.0f}</div>'
                        f'<div style="color:#64748B;font-size:.63rem">{_val_n_matched} trades via Black-Scholes</div></div>'
                        f'<div style="background:#0A2540;border-radius:8px;padding:.55rem .9rem;text-align:center;flex:1">'
                        f'<div style="color:#64748B;font-size:.66rem">USER STRATEGY WIN RATE</div>'
                        f'<div style="color:{_val_u_c};font-size:1.3rem;font-weight:900">{_val_user_correct}/{_val_n}</div>'
                        f'<div style="color:#64748B;font-size:.63rem">({_val_u_pct:.0f}%) — user trades that won</div></div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    _val_df = pd.DataFrame(_val_rows)
                    st.dataframe(
                        _val_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Entry Date":      st.column_config.TextColumn("Entry Date", width="small"),
                            "Exit Date":       st.column_config.TextColumn("Exit Date", width="small"),
                            "AI Signal":       st.column_config.TextColumn("AI Signal", width="small"),
                            "AI Trade":        st.column_config.TextColumn("AI Trade", width="small"),
                            "AI Trade P&L":    st.column_config.TextColumn("AI Trade P&L"),
                            "AI Trade Exit":   st.column_config.TextColumn("AI Exit Reason"),
                            "AI Direction ✓":  st.column_config.TextColumn("AI Direction ✓"),
                            "User Trade":      st.column_config.TextColumn("User Trade", width="small"),
                            "User Trade P&L":  st.column_config.TextColumn("User Trade P&L"),
                            "User Trade Exit": st.column_config.TextColumn("User Exit Reason"),
                            "User Correct":    st.column_config.TextColumn("User Result"),
                            "Outcomes Match":  st.column_config.TextColumn("Outcomes Match"),
                        }
                    )
                    # Footnote for supplemental entries
                    _n_approx = sum(1 for r in _val_rows if r["AI Trade P&L"].endswith(" †"))
                    if _n_approx > 0:
                        st.caption(
                            f"† {_n_approx} entries use AI direction at your DTE ({_ai_cmp_user_dte} days) "
                            f"because the AI-recommended DTE ({_ai_cmp_dte} days) runs out of room "
                            "before the end of the backtest window — those entry dates have no AI comparison otherwise."
                        )

                    # Excel download including the validation comparison sheet
                    try:
                        _vxls_buf = _io.BytesIO()
                        with pd.ExcelWriter(_vxls_buf, engine="openpyxl") as _vxw:
                            _val_df.to_excel(_vxw, sheet_name="AI Signal Validation", index=False)
                            pd.DataFrame(_atm_rows).to_excel(_vxw, sheet_name="User Strategy Trades", index=False)
                            # AI-directed trades sheet
                            _ai_tr_rows = []
                            for _atr in _ai_cmp_trials:
                                _atr_open  = str(_atr.get("openDateTime") or _atr.get("entryDate") or "")[:10]
                                _atr_close = str(_atr.get("closeDateTime") or _atr.get("exitDate") or "")[:10]
                                _atr_pl    = float(_atr.get("profitLoss") or _atr.get("profit_loss") or 0)
                                _atr_rsn   = str(_atr.get("exitReason") or _atr.get("closeReason") or "---").replace("_", " ").title()
                                _ai_tr_rows.append({
                                    "Entry Date":  _atr_open,
                                    "Exit Date":   _atr_close,
                                    "Strategy":    _ai_cmp_action.title(),
                                    "P&L":         f"${_atr_pl:+,.2f}",
                                    "Exit Reason": _atr_rsn,
                                    "Result":      "WIN ✅" if _atr_pl > 0 else "LOSS ❌",
                                })
                            pd.DataFrame(_ai_tr_rows).to_excel(_vxw, sheet_name="AI-Directed BS Trades", index=False)
                            # Summary comparison sheet
                            _ai_wr_pct_fmt = (
                                f"{_ai_cmp_wr * 100:.1f}%" if _ai_cmp_wr <= 1
                                else f"{_ai_cmp_wr:.1f}%"
                            )
                            _sum_cmp = {
                                "Metric": [
                                    "Symbol", "Period", "AI Signal", "AI Trade Direction",
                                    "AI Direction Correct (BS)", "AI Accuracy %",
                                    "AI-Directed BS P&L", "AI-Directed BS Win Rate",
                                    "User Strategy", "User Strategy P&L", "User Win Rate",
                                    "Backtest ID (User)",
                                ],
                                "Value": [
                                    str(symbol),
                                    f"{origin} → {target}",
                                    _ai_sig_lbl,
                                    _ai_cmp_action.title(),
                                    f"{_val_ai_correct}/{_val_n_matched}",
                                    f"{_val_ai_pct:.1f}%",
                                    f"${_ai_cmp_pnl:+,.2f}",
                                    _ai_wr_pct_fmt,
                                    f"{_atm_dir.title()} {_atm_type.title()}",
                                    f"${float(opts.get('profit_loss') or 0):+,.2f}",
                                    f"{_bt_wr_pct:.1f}%",
                                    str(opts.get("backtest_id", "—")),
                                ],
                            }
                            pd.DataFrame(_sum_cmp).to_excel(_vxw, sheet_name="Summary", index=False)
                            # Auto-fit columns on all sheets
                            for _sname, _sws in _vxw.sheets.items():
                                for _vc in _sws.columns:
                                    _vw = max(len(str(_vc_cell.value or "")) for _vc_cell in _vc)
                                    _sws.column_dimensions[_vc[0].column_letter].width = min(_vw + 4, 35)
                        _vxls_buf.seek(0)
                        _vxls_fname = f"{symbol}_AI_Signal_Validation_{origin[:10]}_{target[:10]}.xlsx"
                        st.download_button(
                            label="Download AI Signal Validation Excel",
                            data=_vxls_buf.getvalue(),
                            file_name=_vxls_fname,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="dl_ai_validation_excel",
                        )
                    except Exception:
                        pass

                # Word document export
                try:
                    import docx as _docx_mod
                    from docx.shared import Pt as _Pt, RGBColor as _RGB, Inches as _Inches
                    from docx.enum.text import WD_ALIGN_PARAGRAPH as _WD_ALIGN

                    _doc = _docx_mod.Document()

                    # ── Title ──────────────────────────────────────────────
                    _doc.core_properties.author = "AI Financial Analyst System"
                    _t = _doc.add_heading(
                        f"Backtest Report — {symbol} {_atm_dir.title()} {_atm_type.title()}",
                        level=1,
                    )
                    _t.alignment = _WD_ALIGN.CENTER

                    _sub = _doc.add_paragraph(
                        f"Period: {origin} → {target}   |   "
                        f"DTE: {int(opts_p.get('dte', 0))}   |   "
                        f"Delta: {opts_p.get('delta', '—')}   |   "
                        f"Generated: 2026-08-01"
                    )
                    _sub.alignment = _WD_ALIGN.CENTER

                    _doc.add_paragraph()  # spacer

                    # ── Summary section ────────────────────────────────────
                    _doc.add_heading("Summary", level=2)

                    _sum_rows = [
                        ("Symbol",           str(symbol)),
                        ("Strategy",         f"{_atm_dir.title()} {_atm_type.title()}"),
                        ("Backtest Window",  f"{origin} → {target}"),
                        ("DTE",              str(int(opts_p.get("dte", 0)))),
                        ("Delta",            str(opts_p.get("delta", "—"))),
                        ("Total Trades",     str(_n_total)),
                        ("Wins",             str(sum(1 for r in _atm_rows if "WIN" in str(r.get("Result", ""))))),
                        ("Losses",           str(sum(1 for r in _atm_rows if "LOSS" in str(r.get("Result", ""))))),
                        ("Win Rate",         f"{sum(1 for r in _atm_rows if 'WIN' in str(r.get('Result',''))) / _n_total * 100:.1f}%" if _n_total else "0%"),
                        ("AI Correct",       str(_n_ai_correct)),
                        ("AI Accuracy",      f"{_ai_acc_pct:.1f}%"),
                        ("Total P&L",        f"${float(opts.get('profit_loss') or 0):+,.2f}"),
                        ("Avg P&L / Trade",  f"${float(opts.get('avg_pnl') or 0):+,.2f}"),
                    ]

                    _stbl = _doc.add_table(rows=1, cols=2)
                    _stbl.style = "Light List Accent 1"
                    _stbl.rows[0].cells[0].text = "Metric"
                    _stbl.rows[0].cells[1].text = "Value"
                    for _m, _v in _sum_rows:
                        _row = _stbl.add_row()
                        _row.cells[0].text = _m
                        _row.cells[1].text = _v

                    _doc.add_paragraph()  # spacer

                    # ── Trade-by-trade table (cap at 200 rows to keep doc size sane) ──
                    _doc.add_heading("Trade Details", level=2)
                    _trade_cols = list(_atm_df.columns)
                    _trade_rows_w = _atm_df.head(200).values.tolist()

                    _ttbl = _doc.add_table(rows=1, cols=len(_trade_cols))
                    _ttbl.style = "Light List"
                    for _ci, _ch in enumerate(_trade_cols):
                        _ttbl.rows[0].cells[_ci].text = str(_ch)
                    for _tr in _trade_rows_w:
                        _r = _ttbl.add_row()
                        for _ci, _tv in enumerate(_tr):
                            _r.cells[_ci].text = str(_tv) if _tv is not None else ""

                    if len(_atm_df) > 200:
                        _doc.add_paragraph(
                            f"(Showing first 200 of {len(_atm_df)} trades. Download Excel for full data.)"
                        )

                    # ── AI Analysis section ────────────────────────────────
                    _doc.add_paragraph()
                    _doc.add_heading("AI Signal Analysis", level=2)
                    _doc.add_paragraph(
                        f"AI correctly predicted market direction on {_n_ai_correct} out of {_n_total} trades "
                        f"({_ai_acc_pct:.1f}% accuracy). "
                    )
                    if _ai_acc_pct >= 60:
                        _doc.add_paragraph("Verdict: Strong AI signal — AI direction was reliable for this period/strategy.")
                    elif _ai_acc_pct >= 40:
                        _doc.add_paragraph("Verdict: Mixed AI accuracy — use additional confirmation before trading.")
                    else:
                        _doc.add_paragraph("Verdict: Low AI accuracy for this period — direction signal was unreliable.")

                    # ── Save to buffer ─────────────────────────────────────
                    _docx_buf = _io.BytesIO()
                    _doc.save(_docx_buf)
                    _docx_buf.seek(0)
                    _docx_fname = (
                        f"{symbol}_{_atm_dir}_{_atm_type}_DTE{int(opts_p.get('dte', 0))}"
                        f"_{origin[:10]}_{target[:10]}.docx"
                    )
                    st.download_button(
                        label="Download Trades Word (.docx)",
                        data=_docx_buf.getvalue(),
                        file_name=_docx_fname,
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        key="dl_trades_docx",
                    )
                except Exception as _docx_e:
                    pass

                # AI accuracy summary
                _align_color = "#10B981" if _ai_acc_pct >= 60 else ("#F59E0B" if _ai_acc_pct >= 40 else "#EF4444")
                if _ai_acc_pct >= 60:
                    _acc_verdict = f"AI correctly predicted {_n_ai_correct}/{_n_total} trade outcomes — strong signal"
                elif _ai_acc_pct >= 40:
                    _acc_verdict = f"AI was right on {_n_ai_correct}/{_n_total} trades — mixed accuracy, use with caution"
                else:
                    _acc_verdict = f"AI was right on only {_n_ai_correct}/{_n_total} trades — direction signal unreliable for this period"
                st.markdown(
                    f'<div style="background:#0F172A;border-left:4px solid {_align_color};'
                    f'border-radius:6px;padding:.6rem 1rem;margin:.5rem 0;font-size:.8rem">'
                    f'<span style="color:{_align_color};font-weight:700">AI Accuracy: '
                    f'{_n_ai_correct}/{_n_total} ({_ai_acc_pct:.0f}%)</span>'
                    f'<span style="color:#64748B"> — </span>'
                    f'<span style="color:#CBD5E1">{_acc_verdict}</span></div>',
                    unsafe_allow_html=True,
                )

                # ── HOLD override hypothetical ─────────────────────────────────
                if _ai_is_hold_atm and _raw_trials:
                    _hyp_wins  = sum(1 for t in _raw_trials if float(t.get("profitLoss") or t.get("profit_loss") or 0) > 0)
                    _hyp_total = len(_raw_trials)
                    _hyp_pl    = sum(float(t.get("profitLoss") or t.get("profit_loss") or 0) for t in _raw_trials)
                    _hyp_wr    = (_hyp_wins / _hyp_total * 100) if _hyp_total else 0
                    _hyp_pl_c  = "#10B981" if _hyp_pl >= 0 else "#EF4444"
                    _hyp_wr_c  = "#10B981" if _hyp_wr >= 60 else ("#F59E0B" if _hyp_wr >= 40 else "#EF4444")
                    st.markdown(
                        f'<div style="background:#0A1520;border:1px dashed #F59E0B;border-radius:8px;'
                        f'padding:.8rem 1.1rem;margin:.6rem 0">'
                        f'<div style="color:#FCD34D;font-weight:800;font-size:.83rem;margin-bottom:.5rem">'
                        f'HYPOTHETICAL: What if we overrode HOLD and entered anyway?</div>'
                        f'<div style="display:flex;gap:1.2rem;flex-wrap:wrap;font-size:.78rem">'
                        f'<div style="text-align:center">'
                        f'<div style="color:#64748B;font-size:.68rem">TRADES ENTERED</div>'
                        f'<div style="color:#FCD34D;font-size:1.2rem;font-weight:900">{_hyp_total}</div></div>'
                        f'<div style="text-align:center">'
                        f'<div style="color:#64748B;font-size:.68rem">WIN RATE</div>'
                        f'<div style="color:{_hyp_wr_c};font-size:1.2rem;font-weight:900">{_hyp_wr:.0f}%</div></div>'
                        f'<div style="text-align:center">'
                        f'<div style="color:#64748B;font-size:.68rem">TOTAL P&L</div>'
                        f'<div style="color:{_hyp_pl_c};font-size:1.2rem;font-weight:900">${_hyp_pl:+,.0f}</div></div>'
                        f'</div>'
                        f'<div style="color:#94A3B8;font-size:.72rem;margin-top:.5rem">'
                        f'This shows what the backtester found when it ran all trades — agent said HOLD so '
                        f'it would have entered none of them. Compare to understand what you\'d miss by following HOLD.</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        # ── LOGS TAB ─────────────────────────────────────────────────────────
        with _tt_tab_log:
            _log_t1, _log_t2, _log_t3, _log_t4 = st.tabs(["Trades", "Orders", "Transactions", "Daily Settlement"])

            # ── Trades sub-tab ────────────────────────────────────────────────
            with _log_t1:
                if _raw_trials:
                    import pandas as pd
                    _trade_rows = []
                    for _i, _t in enumerate(_raw_trials, 1):
                        _t_open   = str(_t.get("openDateTime") or _t.get("entryDate") or _t.get("entry_date") or "")[:10]
                        _t_close  = str(_t.get("closeDateTime") or _t.get("exitDate") or _t.get("exit_date") or "")[:10]
                        _t_pl     = float(_t.get("profitLoss") or _t.get("profit_loss") or 0)
                        _t_prem_raw = _t.get("initialPremium") or _t.get("premium") or _t.get("openPremium")
                        _t_fees_raw = _t.get("fees") or _t.get("totalFees")
                        _t_bpr_raw  = _t.get("buyingPowerReduction") or _t.get("buyingPower")
                        _t_roi_raw  = _t.get("returnOnInvestment") or _t.get("roi")
                        _t_reason   = str(_t.get("exitReason") or _t.get("closeReason") or _t.get("exit_reason") or "---").replace("_", " ")
                        _t_prem     = float(_t_prem_raw) if _t_prem_raw is not None else None
                        _t_fees     = float(_t_fees_raw) if _t_fees_raw is not None else None
                        _t_bpr      = float(_t_bpr_raw)  if _t_bpr_raw  is not None else None
                        _t_roi      = float(_t_roi_raw)  if _t_roi_raw  is not None else None
                        _trade_rows.append({
                            "#":            _i,
                            "Opened":       _t_open,
                            "Closed":       _t_close,
                            "Premium":      f"${_t_prem:+,.2f}" if _t_prem is not None else "---",
                            "Fees":         f"${_t_fees:.2f}"   if _t_fees is not None else "---",
                            "Buying Power": f"${_t_bpr:,.2f}"   if _t_bpr  is not None else "---",
                            "Profit/Loss":  f"${_t_pl:+,.2f}",
                            "Close Reason": _t_reason,
                            "ROI":          f"{_t_roi:.2f}%"    if _t_roi  is not None else "---",
                        })
                    df_trades = pd.DataFrame(_trade_rows)
                    st.dataframe(df_trades, use_container_width=True, hide_index=True,
                                 column_config={"#": st.column_config.NumberColumn(width="small")})
                    st.caption(f"Total: {len(_raw_trials)} trade(s). Entries and exits executed at mid price, 15 min before market close.")
                else:
                    st.info("No per-trade data returned by this backtest run.")

            # ── Orders sub-tab ────────────────────────────────────────────────
            with _log_t2:
                _orders_raw = (
                    _res_raw.get("orders")
                    or _res_raw.get("orderHistory")
                    or []
                )
                if _orders_raw:
                    import pandas as pd
                    st.dataframe(pd.DataFrame(_orders_raw), use_container_width=True, hide_index=True)
                elif _raw_trials:
                    # Build synthetic orders table from trial data (matches TastyTrade website Orders format)
                    import pandas as pd
                    from datetime import date as _date_cls
                    _is_bs_ord = opts.get("bs_fallback") or opts.get("bs_source") == "black_scholes_simulation"
                    _ord_dir   = opts_p.get("direction", "Sell").lower()
                    _ord_type  = opts_p.get("opt_type", "Put").lower()
                    _ord_qty   = int(opts_p.get("quantity") or 1)
                    _open_act  = "sell to open"  if _ord_dir == "sell" else "buy to open"
                    _close_act = "buy to close"  if _ord_dir == "sell" else "sell to close"
                    _effect_o  = "credit"        if _ord_dir == "sell" else "debit"
                    _effect_c  = "debit"         if _ord_dir == "sell" else "credit"
                    _orders_synth = []
                    for _oi, _ot in enumerate(_raw_trials, 1):
                        _o_open   = str(_ot.get("openDateTime") or _ot.get("entryDate") or "")[:10]
                        _o_close  = str(_ot.get("closeDateTime") or _ot.get("exitDate") or "")[:10]
                        _o_instr  = str(_ot.get("instrument") or _ot.get("description") or f"{symbol} {_ord_type} option")
                        _o_px_pt  = float(_ot.get("_entry_price_per_point") or _ot.get("openPremium") or 0)
                        _o_px_c   = float(_ot.get("_entry_price_per_point") or _ot.get("openPremium") or 0)
                        _o_cl_pt  = float(_ot.get("_exit_price_per_point") or _ot.get("closePremium") or 0)
                        # Compute DTE at open and close from instrument string
                        _o_expiry = _ot.get("_expiry") or ""
                        _dte_open = _dte_close = "—"
                        try:
                            if _o_expiry and _o_open:
                                _exp_d = _date_cls.fromisoformat(_o_expiry[:10])
                                _opn_d = _date_cls.fromisoformat(_o_open)
                                _dte_open = str((_exp_d - _opn_d).days)
                                if _o_close:
                                    _cls_d = _date_cls.fromisoformat(_o_close)
                                    _dte_close = str((_exp_d - _cls_d).days)
                        except (ValueError, TypeError):
                            pass
                        _legs_open  = f"{_open_act} {_ord_qty} x {_o_instr} ({_dte_open} DTE)" if _dte_open != "—" else f"{_open_act} {_ord_qty} x {_o_instr}"
                        _legs_close = f"{_close_act} {_ord_qty} x {_o_instr} ({_dte_close} DTE)" if _dte_close != "—" else f"{_close_act} {_ord_qty} x {_o_instr}"
                        _orders_synth.append({
                            "Date": _o_open, "Time": "1:15 am IST",
                            "Trade No.": _oi,
                            "Price": (f"{_o_px_pt:.2f}" if _o_px_pt != 0 else "—"),
                            "Effect": _effect_o, "Legs": _legs_open,
                        })
                        if _o_close:
                            _orders_synth.append({
                                "Date": _o_close, "Time": "1:15 am IST",
                                "Trade No.": _oi,
                                "Price": (f"{_o_cl_pt:.2f}" if _o_cl_pt != 0 else "—"),
                                "Effect": _effect_c, "Legs": _legs_close,
                            })
                    if _orders_synth:
                        st.dataframe(pd.DataFrame(_orders_synth), use_container_width=True, hide_index=True)
                        if _is_bs_ord:
                            st.caption("Synthetic orders generated from Black-Scholes computed trades. Prices are theoretical — real market prices will differ.")
                        else:
                            st.caption(
                                "⚠️ Price column shows '—' because the TastyTrade backtester API does NOT return "
                                "individual entry/exit prices — only aggregate trade P&L is available from the API. "
                                "To see actual prices, export the backtest from the TastyTrade website directly."
                            )
                    else:
                        st.info("No order-level data available for this backtest.")
                else:
                    st.info("No order-level data returned by the API for this backtest.")

            # ── Transactions sub-tab ──────────────────────────────────────────
            with _log_t3:
                if _txn_rows:
                    import pandas as pd
                    _txn_display = []
                    for _j, _tx in enumerate(_txn_rows, 1):
                        _txn_display.append({
                            "#":          _j,
                            "Date":       str(_tx.get("date") or _tx.get("Date") or "")[:10],
                            "Time":       str(_tx.get("time") or _tx.get("Time") or "---"),
                            "Trade No.":  _tx.get("tradeNumber") or _tx.get("trade_number") or "---",
                            "Type":       str(_tx.get("type") or _tx.get("transactionType") or "---").replace("_", " "),
                            "Instrument": str(_tx.get("instrument") or _tx.get("description") or "---"),
                            "Price":      f"${float(_tx.get('price') or 0):,.2f}",
                            "Quantity":   _tx.get("quantity") or 1,
                            "Value":      f"${float(_tx.get('value') or _tx.get('amount') or 0):,.2f}",
                            "Effect":     str(_tx.get("effect") or _tx.get("debitCredit") or "---"),
                            "Fees":       f"${float(_tx.get('fees') or 0):.3f}",
                        })
                    st.dataframe(pd.DataFrame(_txn_display), use_container_width=True, hide_index=True)
                else:
                    _txn_auto = []
                    _is_bs_txn = opts.get("bs_fallback") or opts.get("bs_source") == "black_scholes_simulation"
                    _bs_qty_txn = opts_p.get("quantity") or 1
                    _bs_mult_txn = 100
                    for _i2, _t2 in enumerate(_raw_trials, 1):
                        _t2_open  = str(_t2.get("openDateTime") or _t2.get("entryDate") or "")[:10]
                        _t2_close = str(_t2.get("closeDateTime") or _t2.get("exitDate") or "")[:10]
                        _t2_qty   = int(_t2.get("_quantity") or _bs_qty_txn)
                        _t2_instr = str(
                            _t2.get("instrument") or _t2.get("description")
                            or f"{symbol} {str(opts_p.get('opt_type','call')).lower()} option"
                        )
                        # Use BS-computed per-contract prices when available
                        if _is_bs_txn and _t2.get("_entry_price_per_contract") is not None:
                            _entry_px_contract = float(_t2.get("_entry_price_per_contract") or 0)
                            _exit_px_contract  = float(_t2.get("_exit_price_per_contract")  or 0)
                            _open_fees_tx  = float(_t2.get("_open_fees")  or 0)
                            _close_fees_tx = float(_t2.get("_close_fees") or 0)
                            _instr_bs = str(_t2.get("instrument") or _t2_instr)
                            # Re-label instrument with symbol
                            if _instr_bs.startswith("OPT $"):
                                _instr_bs = symbol + " $" + _instr_bs[5:]
                        else:
                            _t2_prem_raw = _t2.get("initialPremium") or _t2.get("premium") or 0
                            _t2_cl_raw   = _t2.get("closePremium")   or _t2.get("exitPremium") or 0
                            _t2_fees_tot = float(_t2.get("fees") or _t2.get("totalFees") or 0)
                            _entry_px_contract = abs(float(_t2_prem_raw)) / max(_t2_qty, 1)
                            _exit_px_contract  = abs(float(_t2_cl_raw))   / max(_t2_qty, 1)
                            _open_fees_tx  = _t2_fees_tot / 2
                            _close_fees_tx = _t2_fees_tot / 2
                            _instr_bs = _t2_instr

                        _is_buy  = opts_p.get("direction", "Buy").lower() in ("buy", "long")
                        _open_type  = "buy to open"  if _is_buy else "sell to open"
                        _close_type = "sell to close" if _is_buy else "buy to close"
                        _open_effect  = "debit"  if _is_buy else "credit"
                        _close_effect = "credit" if _is_buy else "debit"

                        _entry_value = _entry_px_contract * _t2_qty
                        _exit_value  = _exit_px_contract  * _t2_qty
                        _no_price_api = (not _is_bs_txn and _entry_px_contract == 0)

                        _txn_auto.append({
                            "#":          (_i2 * 2) - 1,
                            "Date":       _t2_open,
                            "Time":       "1:15:00 am IST",
                            "Trade No.":  _i2,
                            "Type":       _open_type,
                            "Instrument": _instr_bs,
                            "Price":      ("—" if _no_price_api else f"${_entry_px_contract:,.2f}"),
                            "Quantity":   _t2_qty,
                            "Value":      ("—" if _no_price_api else f"${_entry_value:,.2f}"),
                            "Effect":     _open_effect,
                            "Fees":       ("—" if _no_price_api else f"${_open_fees_tx:.3f}"),
                        })
                        if _t2_close:
                            _txn_auto.append({
                                "#":          _i2 * 2,
                                "Date":       _t2_close,
                                "Time":       "1:15:00 am IST",
                                "Trade No.":  _i2,
                                "Type":       _close_type,
                                "Instrument": _instr_bs,
                                "Price":      ("—" if _no_price_api else f"${_exit_px_contract:,.2f}"),
                                "Quantity":   _t2_qty,
                                "Value":      ("—" if _no_price_api else f"${_exit_value:,.2f}"),
                                "Effect":     _close_effect,
                                "Fees":       ("—" if _no_price_api else f"${_close_fees_tx:.3f}"),
                            })
                    if _txn_auto:
                        import pandas as pd
                        st.dataframe(pd.DataFrame(_txn_auto), use_container_width=True, hide_index=True)
                        if _is_bs_txn:
                            st.caption("Black-Scholes computed prices (theoretical) — TastyTrade API does not return individual transaction prices for this configuration.")
                        else:
                            st.caption(
                                "⚠️ Price / Value / Fees columns show '—' because the TastyTrade backtester API does NOT "
                                "return per-transaction prices. Only aggregate trade P&L is returned by the API. "
                                "Trade dates and P&L are accurate. To see actual prices, export the backtest from "
                                "the TastyTrade website directly."
                            )
                    else:
                        st.info("No transaction data available for this backtest run.")

            # ── Daily Settlement sub-tab ───────────────────────────────────────
            with _log_t4:
                if _daily_rows:
                    import pandas as pd
                    _ds_display = []
                    for _dr2 in _daily_rows:
                        _dr_date = str(_dr2.get("date") or _dr2.get("Date") or "")[:10]
                        _dr_prog = _dr2.get("progress") or _dr2.get("backtestProgress") or "---"
                        _dr_tpl  = float(_dr2.get("totalProfitLoss") or _dr2.get("total_profit_loss") or _dr2.get("Total profit/loss") or 0)
                        _dr_nl   = float(_dr2.get("netLiquidity") or _dr2.get("net_liquidity") or _dr2.get("Net liquidity") or 0)
                        _dr_dd   = float(_dr2.get("drawdown") or _dr2.get("Drawdown") or 0)
                        _dr_roi  = float(_dr2.get("returnOnInvestment") or _dr2.get("roi") or _dr2.get("ROI") or 0)
                        _ds_display.append({
                            "Date":              _dr_date,
                            "Backtest Progress": f"{_dr_prog}%" if isinstance(_dr_prog, (int, float)) else str(_dr_prog),
                            "Total P&L":         f"${_dr_tpl:+,.2f}",
                            "Net Liquidity":     f"${_dr_nl:,.2f}",
                            "Drawdown":          f"{_dr_dd:.2f}%",
                            "ROI":               f"{_dr_roi:.2f}%",
                        })
                    st.dataframe(pd.DataFrame(_ds_display), use_container_width=True, hide_index=True)
                else:
                    # Build calendar-day daily settlement from trial data.
                    # Shows every day from bt_start to bt_end (including weekends, matching TastyTrade).
                    import datetime as _dtm
                    _ds_from_trials = []
                    _used_cap_ds = float(cap) if cap else float(_raw_stats.get("Used capital") or _raw_stats.get("usedCapital") or 0)

                    # Build a map: exit_date → cumulative P&L realized on that day
                    _exit_pl_map: dict = {}
                    for _dt3 in _raw_trials:
                        _exit_d3 = str(_dt3.get("closeDateTime") or _dt3.get("exitDate") or _dt3.get("exit_date") or "")[:10]
                        _pl3 = float(_dt3.get("profitLoss") or _dt3.get("profit_loss") or 0)
                        if _exit_d3:
                            _exit_pl_map[_exit_d3] = _exit_pl_map.get(_exit_d3, 0.0) + _pl3

                    # Determine date range from backtest_range field or fallback to trial dates
                    _br = opts.get("backtest_range", "") or ""
                    _bt_s_str = _br.split("→")[0].strip() if "→" in _br else ""
                    _bt_e_str = _br.split("→")[-1].strip() if "→" in _br else ""
                    # Fallback: use min/max trial dates
                    if not _bt_s_str or not _bt_e_str:
                        _all_open  = [str(_t.get("openDateTime") or _t.get("entryDate") or "")[:10] for _t in _raw_trials if _t.get("openDateTime") or _t.get("entryDate")]
                        _all_close = [str(_t.get("closeDateTime") or _t.get("exitDate") or "")[:10] for _t in _raw_trials if _t.get("closeDateTime") or _t.get("exitDate")]
                        _bt_s_str  = min(_all_open) if _all_open else ""
                        _bt_e_str  = max(_all_close) if _all_close else ""

                    try:
                        _start_dt = _dtm.date.fromisoformat(_bt_s_str)
                        _end_dt   = _dtm.date.fromisoformat(_bt_e_str)
                        _cum_pl   = 0.0
                        _nl_peak  = _used_cap_ds
                        _cur_dt   = _start_dt
                        while _cur_dt <= _end_dt:
                            _d_str   = str(_cur_dt)
                            _day_pl  = _exit_pl_map.get(_d_str, 0.0)
                            _cum_pl += _day_pl
                            _nl_now  = _used_cap_ds + _cum_pl
                            if _nl_now > _nl_peak:
                                _nl_peak = _nl_now
                            _dd_day  = ((_nl_peak - _nl_now) / _nl_peak * 100) if _nl_peak > 0 else 0
                            _roi_day = (_cum_pl / _used_cap_ds * 100) if _used_cap_ds else 0
                            _ds_from_trials.append({
                                "Date":            _d_str,
                                "Total profit/loss": f"${_cum_pl:+,.2f}",
                                "Net liquidity":   f"${_nl_now:,.2f}",
                                "Drawdown":        f"{_dd_day:.2f}%",
                                "ROI":             f"{_roi_day:.2f}%",
                            })
                            _cur_dt += _dtm.timedelta(days=1)
                    except (ValueError, TypeError):
                        pass

                    if _ds_from_trials:
                        import pandas as pd
                        st.dataframe(pd.DataFrame(_ds_from_trials), use_container_width=True, hide_index=True)
                        st.caption("Constructed from trial exit dates — mark-to-market of open positions not included (API limitation).")
                    else:
                        st.info("No daily settlement data available for this backtest run.")

    # ── Underlying Stock Reference (informational only — collapsed) ──────────
    with st.expander("Underlying Stock Price Reference (informational only)", expanded=False):
        st.caption("Not used for options accuracy — shown for context only.")
        if val_ok:
            orig_p = float(val.get("origin_price") or 0)
            tgt_p  = float(val.get("target_price") or 0)
            ret_p  = float(val.get("actual_return_pct") or 0)
            cap_v  = float(val.get("actual_final_capital") or cap)
            pl_v   = float(val.get("actual_total_pl") or 0)
            act_dec_ref = val.get("actual_decision", "---")
            st.markdown(
                _table([
                    ("Underlying Stock Decision", _decision_badge(act_dec_ref)),
                    ("Origin Price",             _fmt(orig_p, "$")),
                    ("Target Price",             _fmt(tgt_p, "$")),
                    ("Actual Stock Return %",    _sign_fmt(ret_p, suffix="%")),
                    ("Stock Final Capital",      _fmt(cap_v, "$")),
                    ("Stock Total P&L",          _sign_fmt(pl_v)),
                ]),
                unsafe_allow_html=True,
            )
        else:
            st.info("Stock validation did not succeed.")

    # ── Final Decision Board ──────────────────────────────────────────────────
    _section_header(None, "Final Decision Board")

    ai_dec = ai.get("decision", "---")

    if opts_status == "SUCCESS":
        # Options backtest succeeded — derive agreement from P&L
        bt_pnl  = float(opts.get("profit_loss") or 0)
        bt_dec  = "BUY" if bt_pnl > 0 else ("SELL" if bt_pnl < 0 else "HOLD")
        ai_dir  = "positive" if ai.get("predicted_return_pct", 0) >= 2 else (
                  "negative" if ai.get("predicted_return_pct", 0) <= -2 else "neutral")
        bt_dir  = "positive" if bt_pnl > 0 else ("negative" if bt_pnl < 0 else "neutral")
        # REVIEW means AI was uncertain — it cannot be a MATCH with any backtest direction
        _ai_is_review = str(ai_dec).upper() == "REVIEW"
        bt_agree = (ai_dir == bt_dir) and not _ai_is_review
        agreement  = "MATCH" if bt_agree else "CONFLICT"
        final_dec  = ai_dec if bt_agree else "REVIEW"
    else:
        agreement  = "BACKTEST_FAILED"
        final_dec  = "REVIEW"
        bt_dec     = "FAILED"

    # Options strategy decision labels
    _opts_label_map   = {"BUY": "ENTER", "SELL": "SKIP", "HOLD": "WAIT", "REVIEW": "REVIEW"}
    _ai_opts_label    = _opts_label_map.get(str(ai_dec).upper(), ai_dec)
    _bt_opts_label    = {
        "BUY": "PROFITABLE", "SELL": "LOST", "HOLD": "FLAT", "FAILED": "FAILED"
    }.get(str(bt_dec).upper(), bt_dec)
    _final_opts_label = _opts_label_map.get(str(final_dec).upper(), final_dec)

    # 5-column board: AI Signal | Trades Executed | P&L Outcome | Agreement | Final Verdict
    _n_trades_fdb = int(opts.get("total_trades") or 0) if opts_status == "SUCCESS" else None
    _wr_fdb       = opts.get("win_rate")
    _wr_fdb_str   = (
        f"{float(_wr_fdb)*100:.0f}%" if (_wr_fdb is not None and float(_wr_fdb) <= 1)
        else (f"{float(_wr_fdb):.0f}%" if _wr_fdb is not None else "---")
    )
    _pnl_fdb      = float(opts.get("profit_loss") or 0) if opts_status == "SUCCESS" else None
    _pnl_clr_fdb  = "#10B981" if (_pnl_fdb is not None and _pnl_fdb >= 0) else "#EF4444"
    _wr_clr_fdb   = (
        "#10B981" if _wr_fdb is not None and (float(_wr_fdb)*100 if float(_wr_fdb)<=1 else float(_wr_fdb)) >= 60
        else "#F59E0B" if _wr_fdb is not None and (float(_wr_fdb)*100 if float(_wr_fdb)<=1 else float(_wr_fdb)) >= 40
        else "#EF4444"
    )
    agree_color_fdb = (
        "#00875A" if agreement == "MATCH"
        else "#DF1B41" if agreement == "CONFLICT"
        else "#D97706"
    )
    _validated_fdb  = agreement == "MATCH"
    _verdict_fdb_color = "#10B981" if _validated_fdb else "#EF4444"
    _verdict_fdb_label = "AGENT VALIDATED" if _validated_fdb else (
        "NOT VALIDATED" if agreement == "CONFLICT" else "REVIEW"
    )

    c5a, c5b, c5c, c5d, c5e = st.columns(5)
    with c5a:
        st.markdown(
            '<div style="background:#0A2540;padding:.6rem .7rem;border-radius:6px;text-align:center">'
            '<div style="color:#8BA9C4;font-size:.68rem;margin-bottom:.3rem">AI STRATEGY SIGNAL</div>'
            f'<div style="font-size:1.5rem;font-weight:900">{_decision_badge(ai_dec)}</div>'
            f'<div style="color:#FCD34D;font-size:.62rem;font-weight:700;margin-top:.15rem">{_ai_opts_label}</div>'
            f'<div style="color:#8BA9C4;font-size:.63rem;margin-top:.1rem">'
            f"Return: {_sign_fmt(ai.get('predicted_return_pct'), suffix='%')}&nbsp;|&nbsp;"
            f"Conf: {ai.get('confidence_score','?')}/100"
            f"</div></div>",
            unsafe_allow_html=True,
        )
    with c5b:
        _trades_disp = (
            f'<div style="font-size:1.8rem;font-weight:900;color:#FCD34D">{_n_trades_fdb}</div>'
            if _n_trades_fdb is not None
            else '<div style="font-size:1rem;font-weight:700;color:#EF4444">---</div>'
        )
        st.markdown(
            '<div style="background:#0A2540;padding:.6rem .7rem;border-radius:6px;text-align:center">'
            '<div style="color:#8BA9C4;font-size:.68rem;margin-bottom:.3rem">TRADES EXECUTED</div>'
            + _trades_disp +
            f'<div style="color:{_wr_clr_fdb};font-size:.7rem;font-weight:700;margin-top:.1rem">'
            f'Win Rate: {_wr_fdb_str}</div>'
            f'<div style="color:#8BA9C4;font-size:.63rem;margin-top:.05rem">'
            f'{opts.get("backtest_range", f"{origin} → {target}")[:28]}</div></div>',
            unsafe_allow_html=True,
        )
    with c5c:
        _pnl_disp = (
            f'<div style="font-size:1.3rem;font-weight:900;color:{_pnl_clr_fdb}">'
            f'${_pnl_fdb:+,.0f}</div>'
            if _pnl_fdb is not None
            else f'<div style="font-size:1rem;font-weight:700;color:#EF4444">{opts_status}</div>'
        )
        st.markdown(
            '<div style="background:#0A2540;padding:.6rem .7rem;border-radius:6px;text-align:center">'
            '<div style="color:#8BA9C4;font-size:.68rem;margin-bottom:.3rem">P&L OUTCOME</div>'
            + _pnl_disp +
            f'<div style="color:#FCD34D;font-size:.62rem;font-weight:700;margin-top:.15rem">{_bt_opts_label}</div>'
            f'<div style="color:#8BA9C4;font-size:.63rem;margin-top:.05rem">'
            f'Avg: {_sign_fmt(opts.get("avg_pnl")) if opts_status=="SUCCESS" else "---"} / trade'
            f'</div></div>',
            unsafe_allow_html=True,
        )
    with c5d:
        st.markdown(
            '<div style="background:#0A2540;padding:.6rem .7rem;border-radius:6px;text-align:center">'
            '<div style="color:#8BA9C4;font-size:.68rem;margin-bottom:.3rem">AGREEMENT</div>'
            f'<div style="font-size:1.3rem;font-weight:900;color:{agree_color_fdb}">{agreement}</div>'
            f'<div style="color:#8BA9C4;font-size:.63rem;margin-top:.2rem">'
            f"{'AI direction ✓ backtest' if agreement=='MATCH' else ('Backtest failed' if agreement=='BACKTEST_FAILED' else 'Directions differ')}"
            f"</div></div>",
            unsafe_allow_html=True,
        )
    with c5e:
        st.markdown(
            f'<div style="background:{"#052E16" if _validated_fdb else "#180A0A"};'
            f'border:2px solid {_verdict_fdb_color};padding:.6rem .7rem;border-radius:6px;text-align:center">'
            '<div style="color:#8BA9C4;font-size:.68rem;margin-bottom:.3rem">FINAL VERDICT</div>'
            f'<div style="font-size:1.6rem;font-weight:900">{_decision_badge(final_dec)}</div>'
            f'<div style="color:{_verdict_fdb_color};font-size:.62rem;font-weight:900;margin-top:.15rem;letter-spacing:.04em">'
            f'{_verdict_fdb_label}</div>'
            f'<div style="color:#8BA9C4;font-size:.63rem;margin-top:.05rem">'
            f"{'Both engines agree' if agreement=='MATCH' else 'Human review needed'}"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    # Verdict narrative
    if agreement == "MATCH":
        st.success(
            f"AGENT VALIDATED — AI predicted {ai_dec} ({_ai_opts_label}), "
            f"backtester ran {_n_trades_fdb} trades with {_wr_fdb_str} win rate and "
            f"${float(opts.get('profit_loss') or 0):+,.2f} total P&L. Directions agree."
        )
    elif agreement == "CONFLICT":
        st.error(
            f"AGENT NOT VALIDATED — AI predicted {ai_dec} ({_ai_opts_label}) but "
            f"backtester P&L was {_bt_opts_label}. Conflict saved for calibration."
        )
    else:
        st.warning(
            f"BACKTEST {opts_status} — {opts.get('error', '')}. "
            "Agreement cannot be determined. Accuracy not saved."
        )

    if saved:
        st.success(f"Options accuracy record saved. {save_msg}")
    else:
        st.info(f"Record not saved: {save_msg}")

    # Known Answer Audit
    if val_ok:
        _render_known_answer_audit(val, cap)

    # ── Auth status + diagnostics (no section header) ────────────────────────
    fu  = ai.get("features_used", {}) or {}
    mom = ai.get("_momentum_signals", {}) or {}

    # ── TASTYTRADE AUTH TRUTH CHECK panel ─────────────────────────────────────
    _tt_truth_d = st.session_state.get("run_context", {}).get("tastytrade_auth_truth", {})
    _tt_allowed = _tt_truth_d.get("backtest_allowed", None)
    _tt_truth_color = (
        "#064E3B" if _tt_allowed is True else
        "#7C2D12" if _tt_allowed is False else "#1E3A5F"
    )
    _tt_truth_label_color = (
        "#10B981" if _tt_allowed is True else
        "#EF4444" if _tt_allowed is False else "#93C5FD"
    )
    _tt_truth_label = (
        "BACKTEST ALLOWED" if _tt_allowed is True else
        "BACKTEST BLOCKED" if _tt_allowed is False else "NOT YET CHECKED"
    )
    if _tt_truth_d:
        st.markdown(
            f'<div style="background:{_tt_truth_color};border:2px solid {_tt_truth_label_color};'
            f'border-radius:8px;padding:.75rem 1.2rem;margin-bottom:.7rem;color:#F8FAFC">'
            f'<div style="color:{_tt_truth_label_color};font-weight:900;font-size:.85rem;margin-bottom:.4rem">'
            f'TASTYTRADE AUTH TRUTH CHECK — {_tt_truth_label}</div>'
            + _table_dark([
                ("Credential Source",         _tt_truth_d.get("credential_source", "---")),
                ("Access Token Present",       str(_tt_truth_d.get("access_token_present", "---"))),
                ("Access Token (masked)",      _tt_truth_d.get("access_token_masked") or "N/A"),
                ("Refresh Token Present",      str(_tt_truth_d.get("refresh_token_present", "---"))),
                ("Refresh Token (masked)",     _tt_truth_d.get("refresh_token_masked") or "N/A"),
                ("Token Refresh Attempted",    str(_tt_truth_d.get("token_refresh_attempted", "---"))),
                ("Token Refresh Status",       _tt_truth_d.get("token_refresh_status", "---")),
                ("Token Refresh HTTP Status",  str(_tt_truth_d.get("token_refresh_http_status") or "N/A")),
                ("Customer Check Attempted",   str(_tt_truth_d.get("customer_check_attempted", "---"))),
                ("Customer Check Status",      _tt_truth_d.get("customer_check_status", "---")),
                ("Auth HTTP Status",           str(_tt_truth_d.get("auth_http_status") or "N/A")),
                ("Backtest Allowed",           _tt_truth_label),
                ("Reason",                     _tt_truth_d.get("reason", "---")),
                ("Secrets Masked",             "YES — no token values exposed"),
                ("Timestamp UTC",              _tt_truth_d.get("timestamp_utc", "---")),
            ]) +
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("Tastytrade Auth Truth Check: not yet run this session (submit an options run to see it).")

    d1, d2 = st.columns(2)
    with d1:
        st.markdown(
            _table([
                ("Run ID",              f"<code>{run_id}</code>"),
                ("Run Timestamp (UTC)", run_ts),
                ("Input Hash",          f'<code style="font-size:.68rem">{hash_v}</code>'),
                ("AI Output Hash",      f'<code style="font-size:.68rem">{ai.get("gemini_output_hash", "---")}</code>'),
                ("Hash Match",          "YES" if ai.get("stock_prediction_input_hash") == hash_v else "NO"),
                ("Leakage Check",       ai.get("leakage_check", "---")),
                ("Bars visible to AI",  f'{fu.get("bars_used", "---")} (cutoff {origin})'),
                ("Last AI bar date",    fu.get("last_ai_bar_date", "---")),
                ("Target price hidden", "YES — not in AI context bars"),
                ("Model version",       ai.get("model_version", "---")),
            ]),
            unsafe_allow_html=True,
        )
    with d2:
        st.markdown(
            _table([
                ("Backtest start",      origin),
                ("Backtest end",        target),
                ("Backtest window",     f"{origin} → {target} (prediction origin to target)"),
                ("Backtest status",     opts_status),
                ("Backtest ID",         opts.get("backtest_id", "N/A")),
                ("Options P&L",         _sign_fmt(opts.get("profit_loss")) if opts_status == "SUCCESS" else "N/A"),
                ("Options win rate",    str(opts.get("win_rate", "N/A")) if opts_status == "SUCCESS" else "N/A"),
                ("Options trades",      str(opts.get("total_trades", "N/A")) if opts_status == "SUCCESS" else "N/A"),
            ]),
            unsafe_allow_html=True,
        )

    _render_decision_distribution_diagnostics()


# ══════════════════════════════════════════════════════════════════════════════
# KNOWN ANSWER AUDIT
# ══════════════════════════════════════════════════════════════════════════════


def _render_known_answer_audit(val: dict, initial_capital: float):
    """
    Optional expander to verify provider prices against known-answer test cases.
    Does NOT affect AI calculation -- only validates the data source.
    """
    with st.expander("Known Answer Audit (Expected vs Provider — click to expand)"):
        st.markdown(
            "<b>Purpose:</b> Enter expected prices to verify provider data matches. "
            "This does NOT change AI prediction -- only validates the data source.",
            unsafe_allow_html=True,
        )

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            exp_origin = st.number_input(
                "Expected Origin Price ($)", min_value=0.0, value=0.0, step=0.01,
                format="%.4f", key="ka_origin",
            )
        with col_b:
            exp_target = st.number_input(
                "Expected Target Price ($)", min_value=0.0, value=0.0, step=0.01,
                format="%.4f", key="ka_target",
            )
        with col_c:
            tolerance_pct = st.number_input(
                "Tolerance (%)", min_value=0.0, max_value=10.0, value=1.0,
                step=0.1, format="%.2f", key="ka_tol",
            )

        if st.button("Verify Known Answer", key="ka_verify"):
            if exp_origin <= 0 or exp_target <= 0:
                st.warning("Enter both expected prices (> 0) before verifying.")
            elif val.get("status") != "SUCCESS":
                st.error("Actual validation did not succeed -- no provider prices to compare.")
            else:
                prov_origin = float(val.get("origin_price") or 0)
                prov_target = float(val.get("target_price") or 0)
                if prov_origin <= 0 or prov_target <= 0:
                    st.error("Provider returned zero/null prices -- cannot compare.")
                else:
                    origin_err_pct    = abs(prov_origin - exp_origin) / exp_origin * 100
                    target_err_pct    = abs(prov_target - exp_target) / exp_target * 100
                    origin_ok         = origin_err_pct <= tolerance_pct
                    target_ok         = target_err_pct <= tolerance_pct
                    exp_return_pct    = (exp_target - exp_origin) / exp_origin * 100
                    exp_final_capital = initial_capital * (exp_target / exp_origin)
                    exp_total_pl      = exp_final_capital - initial_capital

                    st.markdown(
                        _table([
                            ("Expected origin price",             f"${exp_origin:,.4f}"),
                            ("Provider origin price",             f"${prov_origin:,.4f}"),
                            ("Origin diff %",                     _err_color(origin_err_pct, "%")),
                            ("Origin match",                      "YES" if origin_ok else "NO"),
                            ("", ""),
                            ("Expected target price",             f"${exp_target:,.4f}"),
                            ("Provider target price",             f"${prov_target:,.4f}"),
                            ("Target diff %",                     _err_color(target_err_pct, "%")),
                            ("Target match",                      "YES" if target_ok else "NO"),
                            ("", ""),
                            ("Expected return (your prices)",     f"{exp_return_pct:+.6f}%"),
                            ("Expected final capital",            f"${exp_final_capital:,.6f}"),
                            ("Expected total P&L",                f"${exp_total_pl:+,.6f}"),
                            ("Formula",
                             f"${initial_capital:,.2f} x ({exp_target}/{exp_origin}) = ${exp_final_capital:,.6f}"),
                        ]),
                        unsafe_allow_html=True,
                    )
                    if origin_ok and target_ok:
                        st.success(
                            f"PROVIDER DATA VERIFIED -- both prices within {tolerance_pct}% tolerance. "
                            f"Origin error: {origin_err_pct:.4f}%, Target error: {target_err_pct:.4f}%."
                        )
                    else:
                        parts = []
                        if not origin_ok: parts.append(f"origin diff {origin_err_pct:.4f}% > {tolerance_pct}%")
                        if not target_ok: parts.append(f"target diff {target_err_pct:.4f}% > {tolerance_pct}%")
                        st.error(
                            f"PROVIDER PRICE MISMATCH: {'; '.join(parts)}. "
                            "Check price_basis setting or try a different date."
                        )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 -- ACCURACY RECORDS
# ══════════════════════════════════════════════════════════════════════════════


def _render_records():
    _section_header(
        6, "Accuracy Records",
        "stock_prediction_evaluation_runs.jsonl — all records counted, last 20 shown",
    )

    # Refresh button clears session-state record cache so next read hits disk fresh
    _r6col1, _r6col2 = st.columns([5, 1])
    with _r6col2:
        if st.button("Refresh Accuracy", key="refresh_accuracy_btn"):
            st.session_state.pop("_accuracy_records_cache", None)
            st.rerun()

    # File metadata banner
    from pathlib import Path as _P6
    import os as _os6, datetime as _dt6
    _eval_path = _P6(__file__).resolve().parent / "stock_prediction_evaluation_runs.jsonl"
    if _eval_path.exists():
        _fmtime = _dt6.datetime.utcfromtimestamp(_os6.path.getmtime(_eval_path)).strftime("%Y-%m-%dT%H:%M:%SZ")
        _flines = sum(1 for _ in open(_eval_path, encoding="utf-8"))
        with _r6col1:
            st.caption(f"File: `{_eval_path.name}` | Last modified (UTC): `{_fmtime}` | Lines on disk: `{_flines}` | Loaded at: `{_dt6.datetime.utcnow().strftime('%H:%M:%SZ')}`")
    else:
        with _r6col1:
            st.caption("No accuracy file found yet.")

    # Current run badge
    _cur_run_id = st.session_state.get("run_id")
    _saved_flag = st.session_state.get("saved")
    _save_msg   = st.session_state.get("save_msg", "")
    if _cur_run_id:
        if _saved_flag is True:
            st.success(f"Current Run ({_cur_run_id}): SAVED to accuracy log — will appear in table below.")
        elif _saved_flag is False:
            _skip_r = st.session_state.get("opts_result", {}).get("accuracy_skip_reason") or "see run output"
            st.info(f"Current Run ({_cur_run_id}): NOT SAVED — {_skip_r}")
        # else pending/no info

    # Load all records for accurate summary stats, then show last 20 in table
    records = load_stock_prediction_records(limit=10000)
    pending_count = count_pending_predictions()

    summ = get_accuracy_summary(records) if records else {}
    mc = st.columns(5)
    mc[0].metric("Total Validated",  summ.get("total", 0))
    mc[1].metric("Pending Future",   pending_count)
    mc[2].metric("Decision Match %",
                 f"{summ['decision_match_rate']:.1f}%" if summ.get("decision_match_rate") is not None else "---")
    mc[3].metric("Direction Match %",
                 f"{summ['direction_match_rate']:.1f}%" if summ.get("direction_match_rate") is not None else "---")
    mc[4].metric("Avg |Return Error|",
                 f"{summ['avg_abs_return_error']:.2f}pp" if summ.get("avg_abs_return_error") is not None else "---")

    if not records:
        _info_card("No validated accuracy records yet. Complete a historical validation to create the first record.")
        return

    # Check if current run is included in table
    _cur_run_id2 = st.session_state.get("run_id", "")
    _run_ids_in_table = [r.get("run_id", "") for r in records[:20]]
    if _cur_run_id2 and _saved_flag is True:
        if _cur_run_id2 in _run_ids_in_table:
            st.success(f"Current run {_cur_run_id2} IS included in table below.")
        else:
            st.warning(
                f"Current run {_cur_run_id2} was saved but is NOT yet visible in the table. "
                "Click 'Refresh Accuracy' button above to reload from disk."
            )

    import pandas as pd
    rows = []
    for r in records[:20]:
        sp  = r.get("stock_prediction_input") or {}
        ap  = r.get("ai_prediction") or {}
        av  = r.get("actual_validation") or {}
        cmp = r.get("comparison") or {}
        _rid = r.get("run_id", "")
        _cur_marker = " ◀ CURRENT" if _rid == _cur_run_id2 else ""
        rows.append({
            "Timestamp":       (r.get("timestamp") or "")[:19],
            "Run ID":          _rid + _cur_marker,
            "Symbol":          sp.get("symbol", ""),
            "Origin Date":     sp.get("prediction_origin_date", ""),
            "Target Date":     sp.get("target_date", ""),
            "AI Return %":     f"{(ap.get('predicted_return_pct') or 0):+.2f}%",
            "Actual Return %": f"{(av.get('actual_return_pct') or 0):+.2f}%",
            "AI Decision":     ap.get("decision", ""),
            "Actual Decision": av.get("actual_decision", ""),
            "Match":           "YES" if cmp.get("decision_match") else "NO",
            "Ret Err pp":      f"{(cmp.get('return_error_pct') or 0):+.2f}",
            "Cap Error $":     f"${abs(cmp.get('capital_error') or 0):,.0f}",
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR -- ROLLING MONTHLY VALIDATION
# ══════════════════════════════════════════════════════════════════════════════


def _sidebar_rolling():
    with st.sidebar:
        # ── Demo Health Check ──────────────────────────────────────────────────
        st.markdown("### Demo Health Check")
        if st.button("Run Demo Health Check", key="demo_hc_btn", type="primary"):
            _run_demo_health_check()
        st.markdown("---")
        st.markdown(
            "### Rolling Monthly Validation\n"
            "Test multiple past months. "
            "AI predicts each month using only pre-month data. "
            "Actual outcome is checked for that month."
        )
        sym_r   = st.text_input("Symbol",             value="TSLA", key="r_sym")
        ctx_r   = st.text_input("Context Start Date", value="2024-01-01", key="r_ctx")
        n_mo    = st.number_input("Months to validate", value=6, min_value=1, max_value=24, key="r_n")
        cap_r   = st.number_input("Capital ($)", value=50_000, min_value=100, key="r_cap")
        bench_r = st.text_input("Benchmark", value="SPY", key="r_bench")

        if st.button("Run Rolling Validation", key="r_btn", type="secondary"):
            _run_rolling(sym_r.strip().upper(), ctx_r.strip(), int(n_mo), float(cap_r), bench_r.strip().upper())


def _run_demo_health_check():
    """Test Gemini key + RapidAPI + Tastytrade + data symbols. Reports READY / DEGRADED / FAILED."""
    import os as _os
    demo_symbols = ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA"]
    results = {}

    with st.sidebar.status("Running demo health check...", expanded=True):
        # Gemini key check
        gemini_key = _os.getenv("GEMINI_API_KEY", "") or _os.getenv("GOOGLE_API_KEY", "")
        if gemini_key:
            st.sidebar.write(f"✓ Gemini API key present (AI_PROVIDER={_AI_PROVIDER_CFG})")
            gemini_ok = True
        else:
            st.sidebar.write("✗ Gemini API key MISSING — set GEMINI_API_KEY in .env")
            gemini_ok = False

        # RapidAPI health check
        st.sidebar.write("Checking RapidAPI...")
        rapi_hc = _rapidapi_hc()
        if rapi_hc.get("http_status") == 200:
            cnt = rapi_hc.get("total_count", "?")
            st.sidebar.write(f"✓ RapidAPI /market/get-movers — {cnt} movers returned")
            rapidapi_ok = True
        else:
            err_msg = rapi_hc.get("error") or f"HTTP {rapi_hc.get('http_status', 0)}"
            st.sidebar.write(f"✗ RapidAPI failed — {err_msg}")
            rapidapi_ok = False

        # Tastytrade health check
        st.sidebar.write("Checking Tastytrade...")
        tt_hc = _tastytrade_hc()
        if tt_hc.get("customer_verified"):
            st.sidebar.write("✓ Tastytrade /customers/me — token refreshed, customer verified")
            tastytrade_ok = True
        else:
            tt_err = tt_hc.get("error") or "customer_verified=False"
            st.sidebar.write(f"✗ Tastytrade failed — {tt_err}")
            tastytrade_ok = False

        # Data provider check per symbol
        for sym in demo_symbols:
            bars, err = fetch_price_history(sym, min_days=400)
            provider  = get_provider_used(sym)
            if bars and len(bars) >= 100:
                results[sym] = {"ok": True, "bars": len(bars), "provider": provider}
                st.sidebar.write(f"✓ {sym} — {len(bars)} bars [{provider}]")
            else:
                results[sym] = {"ok": False, "error": err or "< 100 bars", "provider": provider}
                st.sidebar.write(f"✗ {sym} — {err or 'insufficient bars'}")

    ok_syms = [s for s, r in results.items() if r["ok"]]
    data_ok = len(ok_syms) >= 3
    paid_ok = rapidapi_ok and tastytrade_ok

    if data_ok and gemini_ok and paid_ok:
        overall = "READY"
        st.sidebar.success(
            f"Demo Status: {overall} — Gemini OK + RapidAPI OK + Tastytrade OK + "
            f"{len(ok_syms)}/{len(demo_symbols)} symbols OK"
        )
    elif ok_syms and gemini_ok:
        overall = "DEGRADED"
        issues = []
        if not rapidapi_ok: issues.append("RapidAPI failed")
        if not tastytrade_ok: issues.append("Tastytrade failed")
        if not issues: issues.append(f"only {len(ok_syms)}/{len(demo_symbols)} symbols OK")
        st.sidebar.warning(f"Demo Status: {overall} — {' | '.join(issues)}")
    else:
        overall = "FAILED"
        st.sidebar.error(f"Demo Status: {overall} — critical checks failed")

    if ok_syms:
        st.sidebar.info(f"Recommended demo symbols: {', '.join(ok_syms)}")


def _run_rolling(symbol, ctx_start, n_months, capital, benchmark):
    today   = date.today()
    results = []

    with st.sidebar.status("Fetching price history..."):
        hist, err = fetch_price_history(symbol, min_days=750)
    if not hist:
        st.sidebar.error(f"Price data unavailable: {err}")
        return

    current = (today.replace(day=1) - timedelta(days=1)).replace(day=1)

    for _ in range(n_months):
        mo_end   = (current + timedelta(days=31)).replace(day=1) - timedelta(days=1)
        origin_s = current.strftime("%Y-%m-%d")
        target_s = mo_end.strftime("%Y-%m-%d")

        try:
            ctx_dt = datetime.strptime(ctx_start, "%Y-%m-%d").date()
            if current <= ctx_dt:
                break
        except Exception:
            break

        spi = {
            "symbol":                        symbol,
            "historical_context_start_date": ctx_start,
            "prediction_origin_date":        origin_s,
            "decision_horizon_days":         (mo_end - current).days,
            "target_date":                   target_s,
            "initial_capital":               capital,
            "benchmark":                     benchmark,
            "validation_mode":               "calendar_month",
        }

        ctx_bars   = filter_history_up_to(hist, origin_s)
        ai_result  = _dispatch_prediction(spi, ctx_bars)
        val_result = run_stock_validation(spi, ai_result, hist)

        if ai_result.get("status") == "SUCCESS" and val_result.get("status") == "SUCCESS":
            save_stock_prediction_record(spi, ai_result, val_result)

        cmp = val_result.get("comparison", {})
        results.append({
            "Month":          origin_s[:7],
            "Origin Date":    origin_s,
            "Target Date":    target_s,
            "Origin Price":   f"${val_result.get('origin_price') or 0:,.2f}",
            "Target Price":   f"${val_result.get('target_price') or 0:,.2f}",
            "AI Pred Price":  f"${ai_result.get('predicted_target_price') or 0:,.2f}",
            "AI Capital":     f"${ai_result.get('predicted_final_capital') or 0:,.2f}",
            "Actual Capital": f"${val_result.get('actual_final_capital') or 0:,.2f}",
            "Return Err":     f"{(cmp.get('return_error_pct') or 0):+.2f}pp",
            "AI Dec":         ai_result.get("decision", "---"),
            "Actual Dec":     val_result.get("actual_decision", "---"),
            "Dir Match":      "YES" if cmp.get("directional_match") else "NO",
            "Dec Match":      "YES" if cmp.get("decision_match") else "NO",
        })
        current = (current - timedelta(days=1)).replace(day=1)

    if results:
        import pandas as pd
        st.sidebar.success(f"Done -- {len(results)} months validated")
        _section_header("", f"Rolling Validation -- {symbol} ({n_months} months)")
        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — ACCURACY CALIBRATION DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════


def _render_calibration_dashboard():
    """Section 7: Accuracy Calibration & Failure Analysis dashboard."""
    import json as _json_cd

    st.markdown("---")
    _section_header(7, "Accuracy Calibration & Failure Analysis")

    DATA_DIR = Path(__file__).resolve().parent / "data"

    def _load_json(path):
        try:
            p = Path(path)
            if p.exists():
                return _json_cd.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
        return None

    batch_sum = _load_json(DATA_DIR / "batch_validation_summary.json")
    fail_json = _load_json(DATA_DIR / "failure_analysis.json")
    cal_json  = _load_json(DATA_DIR / "calibration_profiles.json")
    ba_report = _load_json(DATA_DIR / "calibration_before_after_report.json")

    col1, col2, col3, col4 = st.columns(4)
    if batch_sum:
        col1.metric("Decision Match", f"{batch_sum.get('decision_match_pct', '?')}%")
        col2.metric("Hold Rate", f"{batch_sum.get('hold_rate_pct', '?')}%")
        col3.metric("Direction Match", f"{batch_sum.get('direction_match_pct', '?')}%")
        col4.metric("Avg Return Error", f"{batch_sum.get('avg_return_error_pp', '?')}pp")

        hold_rate = batch_sum.get("hold_rate_pct", 0) or 0
        if hold_rate > 55:
            st.error(f"HOLD bias CRITICAL: {hold_rate}% HOLD rate — model is over-conservative.")
        elif hold_rate > 45:
            st.warning(f"HOLD bias CAUTION: {hold_rate}% HOLD rate — consider threshold adjustment.")

        # Top failure categories
        if batch_sum.get("top_failure_reasons"):
            with st.expander("Top Failure Categories", expanded=True):
                for item in batch_sum["top_failure_reasons"][:8]:
                    st.write(f"**{item['count']}x** {item['reason']}")

        # Per-symbol match
        if batch_sum.get("match_by_symbol"):
            with st.expander("Per-Symbol Decision Match %"):
                sym_data = batch_sum["match_by_symbol"]
                for sym, d in sorted(sym_data.items(), key=lambda x: x[1].get("match_pct", 0)):
                    if d.get("runs", 0) == 0:
                        continue  # skip symbols with no completed runs
                    st.write(f"**{sym}**: {d['match_pct']}% ({d['matches']}/{d['runs']} runs)")

        # Recommendations
        if batch_sum.get("recommended_prompt_changes"):
            with st.expander("Recommended Improvements"):
                for rec in batch_sum["recommended_prompt_changes"]:
                    st.write(f"- {rec}")
    else:
        st.info("No batch summary found. Run batch validation first.")

    # HOLD bias
    if fail_json and fail_json.get("hold_bias"):
        hb = fail_json["hold_bias"]
        with st.expander(f"HOLD Bias Analysis -- {hb.get('severity','?')}"):
            col1, col2 = st.columns(2)
            col1.metric("Overall Hold Rate", f"{hb.get('overall_hold_rate','?')}%")
            col2.metric("False Neutral Rate", f"{hb.get('false_neutral_rate','?')}%")
            if hb.get("warning"):
                st.warning(hb["warning"])
            if hb.get("hold_rate_by_symbol"):
                st.write("Hold rate by symbol:", hb["hold_rate_by_symbol"])

    # Before/after
    if ba_report and not ba_report.get("error"):
        with st.expander("Before/After Calibration Comparison"):
            st.write(f"**Assessment:** {ba_report.get('honest_assessment','N/A')}")
            dm_d  = ba_report.get("decision_match_delta")
            hr_d  = ba_report.get("hold_rate_delta")
            if dm_d is not None:
                st.write(
                    f"Decision Match: {ba_report.get('baseline_decision_match')}% "
                    f"-> {ba_report.get('post_decision_match')}% "
                    f"(delta {dm_d:+.1f}pp)"
                )
            if hr_d is not None:
                st.write(
                    f"Hold Rate: {ba_report.get('baseline_hold_rate')}% "
                    f"-> {ba_report.get('post_hold_rate')}% "
                    f"(delta {hr_d:+.1f}pp)"
                )

    # CLI commands
    with st.expander("CLI Commands to Run Calibration"):
        st.code(
            "# Step 1: Run baseline batch\n"
            "python batch_validation_runner.py --focused --max-runs 50 --tag baseline\n\n"
            "# Step 2: Run failure analysis\n"
            "python failure_analyzer.py --tag baseline\n\n"
            "# Step 3: Build calibration profile\n"
            "python calibration_profile_builder.py --tag baseline\n\n"
            "# Step 4: Run post-calibration batch (after prompt changes)\n"
            "python batch_validation_runner.py --focused --max-runs 50 --tag post_calibration\n"
            "python failure_analyzer.py --tag post_calibration\n"
            "python calibration_profile_builder.py --tag post_calibration\n\n"
            "# Step 5: Compare\n"
            "python compare_calibration_runs.py --baseline baseline --post post_calibration\n\n"
            "# Step 6: Accuracy record hygiene\n"
            "python clean_accuracy_records.py --dry-run",
            language="bash",
        )

    # Download buttons
    col1, col2 = st.columns(2)
    csv_path = DATA_DIR / "batch_validation_results.csv"
    fail_path = DATA_DIR / "failure_analysis.json"
    if csv_path.exists():
        with open(csv_path, "rb") as f:
            col1.download_button("Download Batch CSV", f, "batch_validation_results.csv", "text/csv")
    if fail_path.exists():
        with open(fail_path, "rb") as f:
            col2.download_button("Download Failure JSON", f, "failure_analysis.json", "application/json")


# ══════════════════════════════════════════════════════════════════════════════
# BATCH VALIDATION SECTION — shown in sidebar
# ══════════════════════════════════════════════════════════════════════════════


def _render_batch_validation_sidebar() -> None:
    """Render a collapsed batch validation launcher in the sidebar."""
    with st.sidebar:
        st.markdown("---")
        with st.expander("Batch Validation / Model Improvement", expanded=False):
            st.markdown(
                '<span style="color:#6B7280;font-size:.75rem">'
                "Run AI vs actual comparison across multiple stocks and time windows. "
                "Results saved to data/batch_validation_results.csv"
                "</span>",
                unsafe_allow_html=True,
            )

            bv_symbols = st.text_input(
                "Symbols (comma-separated)",
                value="SPY,TSLA",
                key="bv_symbols",
            )
            bv_horizons = st.text_input(
                "Horizons (days, comma-separated)",
                value="7,14,30",
                key="bv_horizons",
            )
            bv_max_runs = st.number_input(
                "Max runs",
                value=20, min_value=1, max_value=200, step=5,
                key="bv_max_runs",
            )
            bv_months = st.number_input(
                "Months back",
                value=12, min_value=1, max_value=36,
                key="bv_months",
            )
            bv_run = st.button("Run Batch Validation", key="bv_run", type="secondary")

            if bv_run:
                try:
                    from batch_validation_runner import run_batch
                    symbols  = [s.strip().upper() for s in bv_symbols.split(",") if s.strip()]
                    horizons = [int(h.strip()) for h in bv_horizons.split(",") if h.strip()]
                    with st.spinner(f"Running {bv_max_runs} validations ..."):
                        summary = run_batch(
                            symbols     = symbols,
                            horizons    = horizons,
                            max_runs    = int(bv_max_runs),
                            months_back = int(bv_months),
                            verbose     = False,
                        )
                    st.success("Batch complete!")
                    st.metric("Decision Match %", f"{summary.get('decision_match_pct', 0):.1f}%")
                    st.metric("Hold Rate %",      f"{summary.get('hold_rate_pct', 0):.1f}%")
                    st.metric("Avg Return Err pp",
                              f"{summary.get('avg_return_error_pp') or 0:.2f}")
                    if summary.get("top_failure_reasons"):
                        st.markdown("**Top Failure Reasons:**")
                        for item in summary["top_failure_reasons"][:4]:
                            st.markdown(f"- `{item['reason']}` × {item['count']}")
                    if summary.get("recommended_prompt_changes"):
                        st.markdown("**Recommendations:**")
                        for rec in summary["recommended_prompt_changes"][:3]:
                            st.markdown(f"- {rec}")
                    # Download buttons
                    from pathlib import Path as _P
                    csv_f  = _P("data") / "batch_validation_results.csv"
                    json_f = _P("data") / "batch_validation_summary.json"
                    if csv_f.exists():
                        st.download_button(
                            "Download Results CSV",
                            data=csv_f.read_bytes(),
                            file_name="batch_validation_results.csv",
                            mime="text/csv",
                        )
                    if json_f.exists():
                        st.download_button(
                            "Download Summary JSON",
                            data=json_f.read_bytes(),
                            file_name="batch_validation_summary.json",
                            mime="application/json",
                        )
                except ImportError as ie:
                    st.error(f"batch_validation_runner not available: {ie}")
                except Exception as exc:
                    st.error(f"Batch validation error: {exc}")

            # Failure analysis button
            st.markdown("---")
            fa_run = st.button("Analyze Failures", key="fa_run", type="secondary")
            if fa_run:
                try:
                    from failure_analyzer import analyze
                    with st.spinner("Analyzing ..."):
                        analysis = analyze(verbose=False)
                    st.success("Analysis complete!")
                    st.metric("Decision Match %",
                              f"{analysis.get('decision_match_pct', 0):.1f}%")
                    st.metric("Hold Rate %",
                              f"{analysis.get('hold_rate_pct', 0):.1f}%")
                    if analysis.get("recommendations"):
                        st.markdown("**Recommendations:**")
                        for r in analysis["recommendations"][:3]:
                            st.markdown(f"- {r}")
                except ImportError:
                    st.warning("Run batch validation first to generate analysis data.")
                except Exception as exc:
                    st.error(f"Analysis error: {exc}")


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
