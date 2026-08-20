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
    c = {"BUY": "#00875A", "SELL": "#DF1B41", "HOLD": "#D97706"}.get(
        str(d).upper(), "#6B7280"
    )
    return f'<span style="color:{c};font-weight:800;font-size:1.1rem">{d}</span>'


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

    st.markdown(
        '<div style="background:#7F1D1D;padding:.6rem 1.2rem;border-radius:6px;'
        'margin:1rem 0 .3rem 0">'
        '<span style="color:#FEF2F2;font-weight:900;font-size:.95rem">'
        'RUN BLOCKED — DATA / PROVIDER NOT AVAILABLE — OLD OUTPUT CLEARED'
        '</span></div>',
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
        _table([
            ("Next actions", ""),
            ("&nbsp;&nbsp;1", "Retry after network / provider is available"),
            ("&nbsp;&nbsp;2", "Configure a reliable historical data provider (.env)"),
            ("&nbsp;&nbsp;3", "Use another symbol with available data"),
            ("&nbsp;&nbsp;4", "Do NOT interpret any old output above as current run result"),
        ]),
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

    with st.expander("Decision Distribution Diagnostics (HOLD bias check)", expanded=(hold_pct > 45)):
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
        f'<span style="color:#F59E0B;font-weight:900;font-size:.78rem">ACTIVE BUILD: TRUTH_GATE_V4</span>&nbsp;&nbsp;'
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
                '<td style="padding:.12rem .4rem">Live market movers — health check ONLY, NOT historical OHLCV, NOT sent to Gemini</td>'
                '<td style="color:#F59E0B;padding:.12rem .4rem">AVAILABLE (not yet wired into prompt)</td></tr>'
                '<tr><td style="padding:.12rem .4rem">Options chain data</td>'
                '<td style="padding:.12rem .4rem">Live strike/IV for exact-strike selection</td>'
                '<td style="color:#EF4444;padding:.12rem .4rem">NOT WIRED — exact-strike unsupported by TT backtest</td></tr>'
                '<tr><td style="padding:.12rem .4rem">Earnings calendar</td>'
                '<td style="padding:.12rem .4rem">Earnings warning in AI prompt</td>'
                '<td style="color:#EF4444;padding:.12rem .4rem">NOT YET IMPLEMENTED</td></tr>'
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
            "An options position is backtested for the <b>prediction window only</b> "
            "(prediction_origin_date to target_date). "
            "The historical context window is NEVER used for backtesting -- "
            "it is AI study only."
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

    # Section 7 -- Accuracy Calibration Dashboard always visible
    _render_calibration_dashboard()

    # Sidebar rolling validation
    _sidebar_rolling()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 -- STOCK FORM
# ══════════════════════════════════════════════════════════════════════════════


def _render_stock_form():
    _section_header(1, "Prediction Setup", "Stock + dates + capital")

    with st.form("stock_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            symbol = st.text_input(
                "Stock Symbol", value="TSLA",
                help="Real US stock ticker, e.g. TSLA, CVS, MSFT, AAPL",
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

        col4, col5, col6 = st.columns(3)
        with col4:
            ctx_start = st.text_input(
                "Historical Context Start Date", value="2025-07-01",
                help="AI uses price history FROM this date. Must be in the past. (YYYY-MM-DD)",
            )
        with col5:
            origin_dt = st.text_input(
                "Prediction Origin Date", value="2026-03-01",
                help="AI predicts AS OF this date. Must be in the past. (YYYY-MM-DD)",
            )
        with col6:
            horizon = st.number_input(
                "Decision Horizon (days)", value=30, min_value=1, max_value=365,
                help="Days forward from origin date to target date.",
            )

        val_mode = st.selectbox(
            "Validation Mode",
            options=["horizon_days", "calendar_month"],
            help="horizon_days: target = origin + N days. calendar_month: last trading day of origin month.",
        )

        price_basis = st.selectbox(
            "Price Basis",
            options=["close", "open", "high", "low"],
            index=0,
            help="Which price field to use. Default: close (standard for walk-forward validation).",
        )

        try:
            tgt_dt = datetime.strptime(origin_dt.strip(), "%Y-%m-%d") + timedelta(days=int(horizon))
            tgt_str = tgt_dt.strftime("%Y-%m-%d")
        except Exception:
            tgt_str = "---"

        _window_preview(ctx_start, origin_dt, tgt_str, horizon)

        require_strict_coverage = st.checkbox(
            "Require exact historical coverage from requested context start",
            value=False,
            help=(
                "If unchecked (default): run proceeds using available provider data when the requested "
                "context start date is earlier than what the provider has. Clearly labeled as "
                "PROVIDER-CONSTRAINED — AI used effective start date, not your requested date. "
                "If checked: run is BLOCKED when the provider cannot satisfy your requested context start date."
            ),
        )

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

    # ── Strike Selection OUTSIDE form so changing it re-renders immediately ──
    # Inside st.form(), widget changes do not trigger reruns — conditional blocks freeze.
    # Strike Selection must be outside to correctly show/hide Strike vs Delta fields.
    _ss_row = st.columns([1, 1, 1, 1])
    with _ss_row[3]:
        strike_selection = st.selectbox(
            "Strike Selection",
            options=["Delta", "Strike"],
            key="opt_strike_sel",
            help="Delta: select strike by delta value (e.g. 30). Strike: exact price (e.g. 7259, 7570) — no upper cap.",
        )

    with st.form("options_form", clear_on_submit=False):
        # ── Base inputs (same as stock mode) ─────────────────────────────────
        col1, col2, col3 = st.columns(3)
        with col1:
            symbol = st.text_input(
                "Stock Symbol", value="TSLA",
                help="Underlying for both AI prediction and options backtest.",
            )
        with col2:
            benchmark = st.text_input("Benchmark", value="SPY")
        with col3:
            capital = st.number_input(
                "Initial Capital ($)", value=50_000, min_value=100, step=1_000, format="%d",
            )

        col4, col5, col6 = st.columns(3)
        with col4:
            ctx_start = st.text_input(
                "Historical Context Start Date", value="2025-07-01",
                help="AI context window start. NOT used for options backtest.",
            )
        with col5:
            origin_dt = st.text_input(
                "Prediction Origin Date", value="2026-03-01",
                help="Options backtest STARTS here. AI predicts from this date.",
            )
        with col6:
            horizon = st.number_input(
                "Decision Horizon (days)", value=30, min_value=1, max_value=365,
                help="Options backtest ENDS at origin + N days.",
            )

        val_mode = st.selectbox(
            "Validation Mode",
            options=["horizon_days", "calendar_month"],
        )

        price_basis = st.selectbox(
            "Price Basis", options=["close", "open", "high", "low"], index=0,
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
            "Backtest runs from prediction origin date to target date ONLY"
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
                value=1, min_value=1, max_value=100, step=1,
                help="Number of option contracts.",
            )
        # strike_selection is defined OUTSIDE this form (above) so it triggers re-render on change

        # ── Conditional inputs: Exact Strike vs Delta ─────────────────────
        if strike_selection == "Strike":
            es1, es2 = st.columns(2)
            with es1:
                strike_price = st.number_input(
                    "Strike Price",
                    value=0.0, min_value=0.0, max_value=99999.0, step=0.5,
                    help="Exact option strike price. No upper cap — SPY 620, SPX 7570, AAPL 240 all accepted.",
                )
            with es2:
                expiry_date_str = st.text_input(
                    "Expiry Date (YYYY-MM-DD)",
                    value="",
                    placeholder="e.g. 2026-08-15",
                    help="Option expiry date. Required for exact strike validation.",
                )
            es3, _es_pad = st.columns(2)
            with es3:
                dte_val = st.number_input(
                    "DTE (used if Expiry Date is blank)",
                    value=45, min_value=0, max_value=365,
                    help="Days to expiration — only used when Expiry Date field is empty.",
                )
            # proxy delta is ATM (50) by default — not shown to user to avoid confusion
            delta_val = 50
            if strike_price > 0:
                _expiry_show = expiry_date_str if expiry_date_str else f"DTE {dte_val}"
                st.info(
                    f"Strike Mode: **{opt_type} @ {strike_price:g}** | "
                    f"Expiry/DTE: **{_expiry_show}** | "
                    "Large strikes (SPX 7570, SPY 620) are fully supported — no upper cap."
                )
        else:
            # Delta Selection mode (original)
            od1, od2 = st.columns(2)
            with od1:
                delta_val = st.number_input(
                    "Delta (1-99)",
                    value=30, min_value=1, max_value=99,
                    help="Target delta for strike selection (e.g. 30 = 0.30 delta).",
                )
            with od2:
                dte_val = st.number_input(
                    "Expiration (DTE)",
                    value=50, min_value=1, max_value=365,
                    help="Days to expiration at entry.",
                )
            strike_price    = 0.0
            expiry_date_str = ""

        od3, od4 = st.columns(2)
        with od3:
            entry_schedule = st.selectbox(
                "Entry Schedule",
                options=["Once at prediction origin date", "Daily", "Weekly", "Monthly"],
                index=0,
                help="When to enter the position. Default: once at prediction origin date.",
            )
        with od4:
            exit_rule = st.selectbox(
                "Exit Rule",
                options=["Exit at target date", "Exit after N days", "Target profit 50%", "Stop loss 200%"],
                index=0,
                help="When to exit. Default: exit at target date (end of prediction window).",
            )

        # Build summary line — shows actual selected values, never stale defaults
        if strike_selection == "Strike":
            _expiry_label = expiry_date_str if expiry_date_str else f"DTE {dte_val}"
            _strike_display = int(strike_price) if strike_price and strike_price == int(strike_price) else strike_price
            _params_summary = (
                f"{direction} {opt_type}, Strike {_strike_display}, {_expiry_label}, Qty {quantity}"
            )
        else:
            _params_summary = f"{direction} {opt_type}, Delta {delta_val}, DTE {dte_val}, Qty {quantity}"

        st.markdown(
            f'<div style="background:#F0FFF4;border:1px solid #6EE7B7;border-radius:6px;'
            f'padding:.5rem 1rem;font-size:.74rem;color:#065F46;margin:.3rem 0">'
            f'<b>Options Backtest Window:</b>&emsp;'
            f'<code>{origin_dt}</code> to <code>{tgt_str}</code>&emsp;'
            f'<b>({horizon} days)</b>&emsp;&mdash;&emsp;'
            f'{_params_summary}<br>'
            f'<span style="color:#059669">'
            f'Historical context ({ctx_start} to {origin_dt}) is AI study only -- '
            f'backtest NEVER includes historical context window.'
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

        _is_strike_mode = strike_selection == "Strike"
        _is_delta_mode_snap = not _is_strike_mode
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
            "strike_price":                  float(strike_price) if _is_strike_mode else None,
            "expiry_date":                   expiry_date_str.strip() if _is_strike_mode else None,
            "dte":                           int(dte_val),
            "run_id":                        run_id,
            "submitted_at_utc":              datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        _is_delta_mode  = not _is_strike_mode
        _req_strike     = float(strike_price) if (_is_strike_mode and strike_price and float(strike_price) > 0) else None
        _expiry_clean   = expiry_date_str.strip() if expiry_date_str else None

        # ── Single source of truth for all options inputs ─────────────────────
        options_params = {
            "direction":        direction,
            "opt_type":         opt_type,
            "quantity":         int(quantity),
            "strike_selection": strike_selection.lower(),
            "contract_selection_method": "EXACT_STRIKE" if _is_strike_mode else "DELTA_SELECTION",
            # Delta mode fields (null in Strike mode)
            "delta_ui":         int(delta_val) if _is_delta_mode else None,
            "delta_decimal":    round(int(delta_val) / 100.0, 4) if _is_delta_mode else None,
            "delta":            int(delta_val) if _is_delta_mode else None,
            # Strike mode fields (null in Delta mode)
            "requested_strike": _req_strike,
            "strike_price":     _req_strike,
            "expiry_date":      _expiry_clean if _is_strike_mode else None,
            # DTE valid in both modes
            "dte":              int(dte_val),
            "entry_schedule":   entry_schedule,
            "exit_rule":        exit_rule,
        }

        # ── Pre-run validation gate — block invalid inputs before any API call ─
        _pre_errors = []
        if _is_strike_mode and not _req_strike:
            _pre_errors.append(
                f"Strike Mode requires Strike Price > 0. "
                f"You entered: {strike_price!r}. "
                "Examples: SPY=620, SPX=7570, AAPL=240. No upper cap — enter the exact strike."
            )
        if _pre_errors:
            st.session_state["error_msg"] = _pre_errors[0]
            st.session_state["opts_params"] = options_params
            st.session_state["opts_result"] = {
                "status":                "REVIEW_REQUIRED",
                "reason":                _pre_errors[0],
                "backtest_status":       "NOT_RUN",
                "exact_validation_status": "INVALID_INPUT",
                "accuracy_saved":        False,
                "accuracy_skip_reason":  "MISSING_STRIKE_PRICE",
            }
            st.session_state["saved"]    = False
            st.session_state["save_msg"] = "NOT SAVED — strike price missing or zero"
        else:
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

    try:
        ctx_dt   = datetime.strptime(ctx_start, "%Y-%m-%d").date()
        days_req = max(400, (today - ctx_dt).days + 90)
    except Exception:
        days_req = 500

    with st.spinner(f"Fetching historical prices for {symbol}..."):
        hist, hist_err = fetch_price_history(symbol, min_days=days_req)

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

    # Effective date discrepancy — user entered a date the provider can't reach
    _effective_ctx_start = _first_bar
    _date_gap_days = 0
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
            "requested_ctx_start":       ctx_start,
            "first_available_bar":       _effective_ctx_start,
            "coverage_gap_days":         _date_gap_days,
            "coverage_status":           "TRUNCATED",
            "strict_mode":               not allow_provider_truncation,
            "provider_truncation_allowed": allow_provider_truncation,
            "run_allowed":               allow_provider_truncation,
        }
        if not allow_provider_truncation:
            # STRICT MODE: user checked "Require exact historical coverage" — block the run
            _block_msg = (
                f"RUN BLOCKED — STRICT COVERAGE REQUIRED\n\n"
                f"Requested Context Start: {ctx_start}\n"
                f"Provider First Available Date: {_effective_ctx_start}\n"
                f"Gap: {_date_gap_days} days\n\n"
                f"AI Prediction: NOT_RUN\n"
                f"Validation: NOT_RUN\n"
                f"Accuracy Saved: NO\n\n"
                f"Reason: 'Require exact historical coverage' is checked. Provider cannot satisfy the requested "
                f"historical context window. Uncheck 'Require exact historical coverage' to proceed with "
                f"available provider data (PROVIDER-CONSTRAINED run)."
            )
            _run_ctx["input_binding_warning"] = _block_msg
            _run_ctx["validation_status"] = "BLOCKED_DATE_RANGE_UNAVAILABLE"
            _run_ctx["provider_status"]["gemini"] = "NOT_RUN"
            st.session_state["run_context"] = _run_ctx
            st.session_state["error_msg"] = _block_msg
            _log_invalid_run(run_id, input_hash, symbol, "stock",
                             "BLOCKED_DATE_RANGE_UNAVAILABLE",
                             f"User requested {ctx_start}, provider starts {_effective_ctx_start}, gap={_date_gap_days}d. Strict mode blocked.")
            return
        else:
            # PROVIDER-CONSTRAINED MODE (default): warn but allow
            _run_ctx["input_binding_warning"]  = (
                f"PROVIDER-CONSTRAINED RUN — requested context start was {ctx_start}, but provider "
                f"first available bar is {_effective_ctx_start} (gap: {_date_gap_days} days). "
                f"AI used {_effective_ctx_start} as effective context start — NOT your requested {ctx_start}. "
                f"To block this behavior, check 'Require exact historical coverage' in the form."
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

    if ai_result.get("status") != "SUCCESS":
        st.session_state["error_msg"] = (
            f"AI prediction failed: {ai_result.get('error', 'Unknown error')}"
        )
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

    try:
        ctx_dt   = datetime.strptime(ctx_start, "%Y-%m-%d").date()
        days_req = max(400, (today - ctx_dt).days + 90)
    except Exception:
        days_req = 500

    with st.spinner(f"Fetching historical prices for {symbol}..."):
        hist, hist_err = fetch_price_history(symbol, min_days=days_req)

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

    prov_label2 = "Gemini" if _AI_PROVIDER_CFG == "gemini" else "AI"
    with st.spinner(f"Running {prov_label2} prediction (context up to {origin_date})..."):
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
        st.session_state["error_msg"] = (
            f"AI prediction failed: {ai_result.get('error', 'Unknown error')}"
        )
        return

    ai_hash = ai_result.get("stock_prediction_input_hash", "")
    if ai_hash and ai_hash != input_hash:
        st.session_state["error_msg"] = (
            f"Hash mismatch: input_hash={input_hash}, AI hash={ai_hash}."
        )
        return

    # Step 7 -- Options backtest: prediction_origin_date → target_date
    # CRITICAL: start_date = prediction_origin_date, NOT historical_context_start_date
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
        "note":                    "Options backtest window = prediction window only",
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

                    # Build leg based on actual selection mode — NEVER mix strike and delta
                    if _is_exact_mode and _confirmed_strike:
                        # Tastytrade does NOT support fixed-strike backtesting via its API.
                        # Mark exact validation as UNSUPPORTED, then run a delta proxy
                        # labeled clearly as DELTA_PROXY_APPROXIMATE.
                        opts_result["exact_validation_status"] = "UNSUPPORTED_BY_PROVIDER"
                        opts_result["exact_validation_note"] = (
                            "Tastytrade API does not support fixed-strike backtesting. "
                            f"Requested strike {_confirmed_strike} cannot be verified exactly. "
                            "Running delta-proxy (ATM ~50) as reference ONLY — labeled DELTA_PROXY_APPROXIMATE."
                        )
                        opts_result["eligible_for_exact_accuracy"] = False
                        _accuracy_skip_reason = "EXACT_STRIKE_UNSUPPORTED_BY_PROVIDER"
                        # Proxy leg uses delta=50 (ATM reference), clearly not exact
                        leg = {
                            "type":                "equity-option",
                            "direction":           direction_map.get(options_params.get("direction", "Sell"), "short"),
                            "quantity":            options_params.get("quantity", 1),
                            "side":                type_map.get(options_params.get("opt_type", "Put"), "put"),
                            "daysUntilExpiration": _effective_dte_used,
                            "strikeSelection":     "delta",
                            "delta":               50,
                        }
                        opts_result["validation_type"] = "DELTA_PROXY_APPROXIMATE"
                        opts_result["fallback_used"]   = True
                        opts_result["payload_dte"]     = _effective_dte_used
                        opts_result["dte_adjustment_reason"] = "EXPIRY_DATE_PRIORITY" if _confirmed_expiry else "USER_DTE"
                    else:
                        # Delta selection mode — use exactly what the user entered (no silent default)
                        _delta_ui = options_params.get("delta_ui") or options_params.get("delta")
                        if not _delta_ui:
                            opts_result.update({"status": "REVIEW_REQUIRED", "reason": "Delta value missing in Delta mode — cannot proceed."})
                            return opts_result
                        leg = {
                            "type":                "equity-option",
                            "direction":           direction_map.get(options_params.get("direction", "Sell"), "short"),
                            "quantity":            options_params.get("quantity", 1),
                            "side":                type_map.get(options_params.get("opt_type", "Put"), "put"),
                            "daysUntilExpiration": _effective_dte_used,
                            "strikeSelection":     "delta",
                            "delta":               int(_delta_ui),
                        }
                        opts_result["payload_dte"] = _effective_dte_used
                        opts_result["validation_type"] = "DELTA_SELECTION"
                        opts_result["fallback_used"]   = False
                        opts_result["eligible_for_exact_accuracy"] = False

                    payload = _tt_build_legs(
                        symbol=symbol,
                        start_date=origin_date,
                        end_date=target_date,
                        legs=[leg],
                    )
                    st.session_state["backtest_payload"] = payload.to_dict()
                    backtest_id, err = _tt_create_backtest(payload)
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
                            _avg_pnl = float(
                                stats.get("Avg. profit/loss per trade")
                                or stats.get("Avg. return per trade")
                                or stats.get("avgProfitLoss") or stats.get("avg_pnl") or 0
                            )
                            _n_trades = int(
                                stats.get("Number of trades") or stats.get("numTrades")
                                or stats.get("total_trades") or len(_trials) or 0
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
                                opts_result.update({
                                    "status":       "SUCCESS",
                                    "backtest_id":  backtest_id,
                                    "win_rate":     _win_pct,
                                    "profit_loss":  _total_pl,
                                    "avg_pnl":      _avg_pnl,
                                    "total_trades": _n_trades,
                                    "raw_stats":    stats,
                                })
                        else:
                            opts_result["status"] = "BACKTEST_POLL_FAILED"
                            opts_result["error"]  = poll_err
                    else:
                        opts_result["status"] = "BACKTEST_CREATE_FAILED"
                        opts_result["error"]  = err
                except Exception as exc:
                    opts_result["status"] = "ERROR"
                    opts_result["error"]  = f"{type(exc).__name__}: {exc}"
    else:
        opts_result["error"] = "Tastytrade backtester not available (import failed or credentials missing)"

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
        _err_card(f"AI prediction failed: {ai.get('error', 'Unknown')}")
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
            ]),
            unsafe_allow_html=True,
        )

    _warn_card(
        f"<b>Data Leakage Prevention:</b> AI received bars from <b>{ctx_start}</b> to "
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
        "rapidapi_tradingview": "RapidAPI TradingView",
        "external_historical_provider": "External Historical Provider (NASDAQ public)",
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
        f'Historical Bars: <b>{_etp_hist_label}</b><br>'
        f'Historical Source: <b>{_etp_hist_src}</b></div>'
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
                ("Data Quality Score",    f"{_dq}/100" if _dq is not None else "---"),
                ("Effective Origin Date", ai.get("effective_origin_date", "---")),
                ("AI Provider",           _ai_prov.upper()),
                ("Gemini Used",           _gem_label),
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
                    ("Historical Price Provider", "Historical price data provider"),
                ]),
                unsafe_allow_html=True,
            )
        elif val_ok:
            _prov = get_provider_used(spi.get("symbol", ""))
            _prov_label = "RapidAPI TradingView" if _prov == "rapidapi_tradingview" else "Historical price data provider"
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

    # AI reasoning + leakage
    if ai.get("reasoning"):
        _info_card(f"<b>AI Reasoning:</b> {ai['reasoning']}")

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

    # ── Section 9 — Developer Debug ──────────────────────────────────────────
    _section_header(9, "Developer Debug", "Raw JSON, hash proof, momentum signals")
    fu  = ai.get("features_used", {}) or {}
    mom = ai.get("_momentum_signals", {}) or {}
    _gem_used_d  = bool(ai.get("gemini_used"))
    _ai_prov_d   = ai.get("ai_provider", "baseline")
    _fp_d        = ai.get("_feature_packet", {}) or {}
    _cal_d       = ai.get("_calibration_summary", {}) or {}
    _gem_raw_d   = ai.get("_gemini_raw_json", {}) or {}

    c9a, c9b = st.columns(2)
    with c9a:
        st.markdown(
            _table([
                ("Run ID",                f"<code>{run_id}</code>"),
                ("Run Timestamp (UTC)",   run_ts),
                ("Input Hash",            f'<code style="font-size:.68rem">{hash_v}</code>'),
                ("AI Output Hash",        f'<code style="font-size:.68rem">{ai.get("stock_prediction_input_hash", "---")}</code>'),
                ("Hash Match",            "YES" if ai.get("stock_prediction_input_hash") == hash_v else "NO"),
                ("Leakage Check",              ai.get("leakage_check", "---")),
                ("Bars visible to AI",         f'{fu.get("bars_used", "---")} (cutoff {origin})'),
                ("Last AI bar date",           fu.get("last_ai_bar_date", "---")),
                ("Target price hidden",        "YES — not in AI context"),
                ("actual_provider_used",       "external_historical_provider"),
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

    with st.expander("User Input Snapshot vs Effective Input (date binding proof)"):
        _snap = st.session_state.get("user_input_snapshot", {})
        _rctx = st.session_state.get("run_context", {})
        _ibw2 = _rctx.get("input_binding_warning")
        if _ibw2:
            st.error(_ibw2)
        _uis1, _uis2 = st.columns(2)
        with _uis1:
            st.markdown("**User-Entered Inputs (exact, captured at submit)**")
            st.json(_snap)
        with _uis2:
            st.markdown("**Effective Inputs Used by AI (after provider constraints)**")
            st.json({
                "effective_ctx_start":    _rctx.get("effective_ctx_start") or _snap.get("historical_context_start_date", "same as requested"),
                "requested_ctx_start":    _rctx.get("requested_ctx_start") or _snap.get("historical_context_start_date"),
                "ctx_start_gap_days":     _rctx.get("ctx_start_gap_days", 0),
                "provider_status":        _rctx.get("provider_status", {}),
                "input_binding_warning":  _ibw2 or "NONE — inputs matched",
            })
    with st.expander("Input JSON"):
        st.json(spi)
    with st.expander("AI visible history summary (first / last 3 bars)"):
        ctx_bars_all = [b for b in hist if ctx_start <= b["date"] <= origin]
        st.json({
            "n_bars":     len(ctx_bars_all),
            "first_3":    ctx_bars_all[:3],
            "last_3":     ctx_bars_all[-3:],
            "ctx_start":  ctx_start,
            "cutoff":     origin,
            "target":     target,
            "note":       "target-date price was NOT in these bars",
        })
    with st.expander("AI Prediction Raw JSON"):
        st.json({k: v for k, v in ai.items()
                 if k not in ("_momentum_signals", "_feature_packet", "_calibration_summary", "_gemini_raw_json")})
    if _gem_used_d:
        with st.expander("Gemini Feature Packet JSON"):
            st.json(_fp_d)
        with st.expander("Gemini Raw Response JSON"):
            st.json(_gem_raw_d)
        with st.expander("Gemini Calibration Summary"):
            st.json(_cal_d)
    with st.expander("Actual Validation Raw JSON"):
        st.json({k: v for k, v in val.items() if k != "comparison"} if val else {})
    with st.expander("Comparison Raw JSON"):
        st.json(cmp)
    with st.expander("Momentum Signals Used by AI"):
        st.json(fu)
    _sig = ai.get("signal_scores", {})
    if _sig:
        with st.expander("Signal Score Engine (bull/bear/uncertainty)"):
            _bull = _sig.get("bullish_score", 0)
            _bear = _sig.get("bearish_score", 0)
            _unc  = _sig.get("uncertainty_score", 0)
            _dom  = _sig.get("dominant", "---")
            _doms = _sig.get("dominant_score", 0)
            st.markdown(
                _table([
                    ("Bullish Score",    f'<b style="color:#10B981">{_bull}/100</b>'),
                    ("Bearish Score",    f'<b style="color:#EF4444">{_bear}/100</b>'),
                    ("Uncertainty Score",f'<b style="color:#F59E0B">{_unc}/100</b>'),
                    ("Dominant Signal",  f'<b>{_dom.upper()} ({_doms}/100)</b>'),
                ]),
                unsafe_allow_html=True,
            )

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
        _err_card(f"AI prediction failed: {ai.get('error', 'Unknown')}")
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

    # ── Section 2 — Historical Context ───────────────────────────────────────
    _section_header(
        2, "Historical Context Used by AI",
        f"Bars {ctx_start} to {origin} — AI study only. Backtest uses prediction window only.",
    )
    n_ctx  = len([b for b in hist if ctx_start <= b["date"] <= origin])
    ctx_ok = ctx_summ.get("status") == "OK"

    c2a, c2b = st.columns([3, 2])
    with c2a:
        st.markdown(
            _table([
                ("Symbol",                    symbol),
                ("Historical context window", f"{ctx_start}  →  {origin}"),
                ("Prediction / validation window", f"{origin}  →  {target}  ({horizon} days)"),
                ("Last bar visible to AI",     max((b["date"] for b in hist if b["date"] <= origin), default="---")),
                ("Bars passed to AI",          f"{n_ctx} trading days"),
                ("Context start price",        _fmt(ctx_summ.get("start_price"), "$") if ctx_ok else "---"),
                ("Origin price (AI's last)",   _fmt(ctx_summ.get("end_price"), "$") if ctx_ok else "---"),
                ("Context period return",      _fmt(ctx_summ.get("return_pct"), suffix="%") if ctx_ok else "---"),
            ]),
            unsafe_allow_html=True,
        )
    with c2b:
        _warn_card(
            "<b>Two-Window Architecture</b><br>"
            f"<b>Context window</b> ({ctx_start} → {origin}): AI study only.<br>"
            f"<b>Prediction window</b> ({origin} → {target}): AI predicts + backtest checks.<br>"
            "<b>Backtest start = prediction_origin_date.</b><br>"
            "Context window is NEVER used as backtest start."
        )
        if ctx_ok:
            st.metric("Origin Price (AI's last known)", f"${ctx_summ.get('end_price', 0):,.2f}")

    # ── Section 2B — Inputs Summary Table ────────────────────────────────────
    _section_header("2B", "Inputs Used")
    i1, i2 = st.columns(2)
    with i1:
        st.markdown(
            _table([
                ("Symbol",              symbol),
                ("Context Start",       ctx_start),
                ("Prediction Origin",   origin),
                ("Target Date",         target),
                ("Capital",             _fmt(cap, "$")),
                ("Benchmark",           spi.get("benchmark", "---")),
                ("Price Basis",         spi.get("price_basis", "close")),
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
                ("Exit Rule",          opts_p.get("exit_rule", "---")),
                ("Backtest Window",    f"{origin}  →  {target}"),
            ]),
            unsafe_allow_html=True,
        )

    # ── Paid API Usage Proof (options mode) — single truth source ────────────
    # RapidAPI: from session_state["rapidapi_health"] (real health check)
    # Tastytrade: from run_context["tastytrade_auth_truth"] ONLY (never from stale health)
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

    st.markdown(
        f'<div style="background:#0A1628;border:1px solid #2563EB;'
        f'border-radius:8px;padding:.7rem 1.1rem;margin:.6rem 0">'
        f'<div style="color:#93C5FD;font-weight:800;font-size:.82rem;margin-bottom:.5rem">'
        f'PAID API USAGE PROOF — Run {_current_rid or "N/A"}</div>'
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:.3rem 1.2rem;font-size:.71rem">'
        f'<div style="color:#FCD34D;font-weight:700;margin-bottom:.2rem">RAPIDAPI (TradingView)</div>'
        f'<div style="color:#FCD34D;font-weight:700;margin-bottom:.2rem">TASTYTRADE (OAuth) — AUTH TRUTH</div>'
        f'<div style="color:#F1F5F9">'
        f'API Key Present: {_ab(_ro_key_ok)}<br>'
        f'Called This Run: {_ab(_ro_called)}<br>'
        f'HTTP Status: <b>{_ro_status if _ro_called else "---"}</b><br>'
        f'Endpoint: <code style="font-size:.66rem">{_ro_endpoint.replace("https://","")}</code><br>'
        f'Total Movers: <b>{_ro_count if _ro_count is not None else "---"}</b><br>'
        f'Top Symbols: <b>{", ".join(_ro_syms[:3]) if _ro_syms else "---"}</b><br>'
        f'Role: <b>LIVE_MARKET_MOVERS / PROVIDER_HEALTH</b><br>'
        f'Used for Historical OHLCV: {_ab(False, "YES", "NO")}<br>'
        f'Used for Stock Validation: {_ab(False, "YES", "NO")}<br>'
        f'Used for Gemini Historical Prompt: {_ab(_ro_leakage, "YES", "NO")}<br>'
        f'Data To Gemini: {_ab(not _ro_leakage, "NO (leakage-safe)", "YES (LEAK!)")}<br>'
        f'Used In Stock Mode: {_ab(True, "YES — health check only", "NO")}<br>'
        f'Used In Options Mode: {_ab(True, "YES — health check only", "NO")}<br>'
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
        + f'<br><b style="color:#FCD34D">OPTIONS BACKTEST:</b><br>'
        f'Backtest Status: <b style="color:{"#10B981" if opts_status == "SUCCESS" else "#EF4444"}">'
        f'{opts_status}</b><br>'
        f'Backtest ID: <b>{opts.get("backtest_id") or "—"}</b><br>'
        + (
            f'Options P&L: <b style="color:{"#10B981" if float(opts.get("profit_loss",0) or 0)>=0 else "#EF4444"}">'
            f'${float(opts.get("profit_loss",0) or 0):+,.2f}</b><br>'
            f'Win Rate: <b>{f"{float(opts.get("win_rate",0) or 0)*100:.1f}%"}</b><br>'
            f'Trials / Trades: <b>{opts.get("total_trades","---")}</b><br>'
            if opts_status == "SUCCESS" else
            f'P&L: <b>N/A</b><br>Win Rate: <b>N/A</b><br>Trades: <b>N/A</b><br>'
        )
        + f'Accuracy Saved: {_ab(st.session_state.get("saved", False))}<br>'
        + (f'Auth/Backtest Reason: <span style="color:#FCA5A5;font-size:.67rem">{_tt_reason_ui}</span>' if not _tt_allowed_ui else "")
        + (f'<br>Backtest Error: <span style="color:#EF4444">{opts.get("error","")}</span>' if opts_status not in ("SUCCESS", "SKIPPED") else "")
        + f'</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Section 3 (left) | Section 4 (right) — options side-by-side ──────────
    st.markdown(
        f'<div style="color:#8BA9C4;font-size:.72rem;margin:.8rem 0 .2rem 0">'
        f'<b>{symbol}</b> &nbsp;|&nbsp; {origin} → {target} &nbsp;({horizon} days)'
        f'</div>',
        unsafe_allow_html=True,
    )

    c3l, c3r = st.columns(2)

    with c3l:
        st.markdown(
            '<div style="background:#0F2940;padding:.4rem 1rem;border-radius:6px;'
            'color:#93C5FD;font-weight:800;font-size:.82rem;margin-bottom:.4rem">'
            "SECTION 3 — AI Options Strategy Prediction</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            _table([
                ("AI Decision",             _decision_badge(ai.get("decision", "---"))),
                ("Origin Price Used",        _fmt(ai.get("origin_price_used"), "$")),
                ("Predicted Target Price",   _fmt(ai.get("predicted_target_price"), "$")),
                ("Predicted Return %",       _sign_fmt(ai.get("predicted_return_pct"), suffix="%")),
                ("Predicted Final Capital",  _fmt(ai.get("predicted_final_capital"), "$")),
                ("Predicted Total P&L",      _sign_fmt(ai.get("predicted_total_pl"))),
                ("Confidence Score",         f"{ai.get('confidence_score', '---')}/100"),
                ("Risk Score",               f"{ai.get('risk_score', '---')}/100"),
                ("Leakage Check",            ai.get("leakage_check", "---")),
            ]),
            unsafe_allow_html=True,
        )

    with c3r:
        st.markdown(
            '<div style="background:#052E1A;padding:.4rem 1rem;border-radius:6px;'
            'color:#6EE7B7;font-weight:800;font-size:.82rem;margin-bottom:.4rem">'
            "SECTION 4 — Options Backtest Actual</div>",
            unsafe_allow_html=True,
        )
        if opts_status == "SUCCESS":
            pnl        = opts.get("profit_loss")
            win_rate   = opts.get("win_rate")
            avg_pnl    = opts.get("avg_pnl")
            tot_trades = opts.get("total_trades")
            wr_fmt = (
                f"{float(win_rate)*100:.1f}%" if (win_rate is not None and float(win_rate) <= 1)
                else (f"{float(win_rate):.1f}%" if win_rate is not None else "---")
            )
            st.markdown(
                _table([
                    ("Backtest Status",   "SUCCESS"),
                    ("Backtest Window",   f"{origin}  →  {target}"),
                    ("Total P&L",         _sign_fmt(pnl) if pnl is not None else "---"),
                    ("Win Rate",          wr_fmt),
                    ("Avg P&L / Trade",   _sign_fmt(avg_pnl) if avg_pnl is not None else "---"),
                    ("Total Trades",      str(tot_trades) if tot_trades is not None else "---"),
                    ("Backtest ID",       opts.get("backtest_id", "---")),
                ]),
                unsafe_allow_html=True,
            )
        elif opts_status == "SKIPPED":
            _warn_card(
                f"<b>Options backtest skipped.</b><br>"
                f"{opts.get('error', opts.get('note', 'Tastytrade backtester unavailable.'))}.<br>"
                "Check TASTYTRADE_USERNAME / TASTYTRADE_PASSWORD in .env."
            )
        elif opts_status == "FLAT_NO_TRADES":
            _err_card(
                f"<b>Tastytrade authentication succeeded. Backtest returned zero trades.</b><br>"
                "This is a <b>parameter / contract availability</b> issue — not a credential issue.<br>"
                "Possible causes: exact strike unsupported by provider, "
                f"strike too far from underlying price ({opts.get('strike_distance_warning', '')}), "
                "DTE/expiry mismatch, no matching contracts in the date window, "
                "or delta proxy found no trades. Check strike distance and DTE/expiry below."
            )
        else:
            _err_card(
                f"<b>Options backtest {opts_status}.</b><br>"
                f"{opts.get('error', 'Unknown error')}."
            )

    if ai.get("reasoning"):
        _info_card(f"<b>AI Reasoning:</b> {ai['reasoning']}")

    # ── Section 3B — Options Strategy Parameters ─────────────────────────────
    _section_header(
        "3B", "Options Strategy Parameters",
        f"Backtest: {origin} → {target}  (prediction window ONLY — never context window)",
    )
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
                ("Entry Schedule",  opts_p.get("entry_schedule", "---")),
                ("Exit Rule",       opts_p.get("exit_rule", "---")),
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

        ob1, ob2 = st.columns(2)
        with ob1:
            st.markdown("**AI OPTIONS PREDICTION**")
            st.markdown(
                _table([
                    ("AI Decision",       _decision_badge(ai.get("decision", "---"))),
                    ("Predicted Return %", _sign_fmt(ai.get("predicted_return_pct"), suffix="%")),
                    ("Confidence",         f"{ai.get('confidence_score','---')}/100"),
                    ("Risk Score",         f"{ai.get('risk_score','---')}/100"),
                ]),
                unsafe_allow_html=True,
            )
        with ob2:
            st.markdown("**TASTYTRADE OPTIONS BACKTEST ACTUAL**")
            st.markdown(
                _table([
                    ("Backtest Status",  "SUCCESS"),
                    ("Total P&L",        f"${pnl_v:+,.2f}"),
                    ("Win Rate",         wr_str),
                    ("Avg P&L / Trade",  f"${avg_pnl_v:+,.2f}"),
                    ("Number of Trades", str(n_trades_v)),
                    ("Backtest ID",      opts.get("backtest_id", "---")),
                    ("Backtest Window",  f"{origin}  →  {target}"),
                ]),
                unsafe_allow_html=True,
            )
    else:
        _err_card(
            f"<b>Options Backtest {opts_status} — cannot compare AI vs backtest.</b><br>"
            f"{opts.get('error', 'Unknown error')}.<br>"
            "<b>Comparison and accuracy save are BLOCKED until backtest succeeds.</b>"
        )

    # ── Underlying Stock Reference (informational only) ───────────────────────
    _section_header(
        None, "Underlying Stock Reference",
        "INFORMATIONAL ONLY — NOT used for options agreement or accuracy",
    )
    st.markdown(
        '<div style="background:#1E293B;border-left:4px solid #F59E0B;'
        'padding:.5rem 1rem;border-radius:4px;font-size:.73rem;color:#FCD34D;margin:.3rem 0">'
        'The underlying stock price movement is shown here for context only. '
        'It does NOT drive the options final decision or accuracy record. '
        'Options accuracy requires a successful Tastytrade options backtest.</div>',
        unsafe_allow_html=True,
    )
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
                ("Note",                     "Reference only — not options accuracy"),
            ]),
            unsafe_allow_html=True,
        )
    else:
        _warn_card("Underlying stock reference unavailable — stock validation did not succeed.")

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
        bt_agree = ai_dir == bt_dir
        agreement  = "MATCH" if bt_agree else "CONFLICT"
        final_dec  = ai_dec if bt_agree else "REVIEW"
    else:
        agreement  = "BACKTEST_FAILED"
        final_dec  = "REVIEW"
        bt_dec     = "FAILED"

    # Options strategy decision labels (ENTER / SKIP / WAIT / REVIEW)
    _opts_label_map = {"BUY": "ENTER", "SELL": "SKIP", "HOLD": "WAIT", "REVIEW": "REVIEW"}
    _ai_opts_label  = _opts_label_map.get(str(ai_dec).upper(), ai_dec)
    _bt_opts_label  = {"BUY": "ENTER_WORKED", "SELL": "STRATEGY_LOST", "HOLD": "FLAT", "FAILED": "FAILED"}.get(str(bt_dec).upper(), bt_dec)
    _final_opts_label = {"BUY": "ENTER", "SELL": "SKIP", "HOLD": "WAIT", "REVIEW": "REVIEW"}.get(str(final_dec).upper(), final_dec)

    c5a, c5b, c5c, c5d = st.columns(4)
    with c5a:
        st.markdown(
            '<div style="background:#0A2540;padding:.6rem 1rem;border-radius:6px;text-align:center">'
            '<div style="color:#8BA9C4;font-size:.7rem;margin-bottom:.3rem">AI STRATEGY SIGNAL</div>'
            f'<div style="font-size:1.6rem;font-weight:900">{_decision_badge(ai_dec)}</div>'
            f'<div style="color:#8BA9C4;font-size:.62rem;margin-top:.15rem">Options: <b style="color:#FCD34D">{_ai_opts_label}</b></div>'
            f'<div style="color:#8BA9C4;font-size:.66rem;margin-top:.1rem">'
            f"Return: {_sign_fmt(ai.get('predicted_return_pct'), suffix='%')}"
            f"</div></div>",
            unsafe_allow_html=True,
        )
    with c5b:
        _bt_badge = (
            f'<span style="color:#EF4444;font-weight:900">FAILED</span>'
            if opts_status != "SUCCESS" else _decision_badge(bt_dec)
        )
        _bt_note = (
            f"P&L: ${float(opts.get('profit_loss') or 0):+,.2f}"
            if opts_status == "SUCCESS" else opts_status
        )
        st.markdown(
            '<div style="background:#0A2540;padding:.6rem 1rem;border-radius:6px;text-align:center">'
            '<div style="color:#8BA9C4;font-size:.7rem;margin-bottom:.3rem">OPTIONS BACKTEST</div>'
            f'<div style="font-size:1.6rem;font-weight:900">{_bt_badge}</div>'
            f'<div style="color:#8BA9C4;font-size:.62rem;margin-top:.15rem">Result: <b style="color:#FCD34D">{_bt_opts_label}</b></div>'
            f'<div style="color:#8BA9C4;font-size:.66rem;margin-top:.1rem">{_bt_note}</div></div>',
            unsafe_allow_html=True,
        )
    with c5c:
        agree_color = (
            "#00875A" if agreement == "MATCH"
            else ("#DF1B41" if agreement == "CONFLICT"
            else "#D97706")
        )
        st.markdown(
            '<div style="background:#0A2540;padding:.6rem 1rem;border-radius:6px;text-align:center">'
            '<div style="color:#8BA9C4;font-size:.7rem;margin-bottom:.3rem">AGREEMENT</div>'
            f'<div style="font-size:1.4rem;font-weight:900;color:{agree_color}">{agreement}</div>'
            f'<div style="color:#8BA9C4;font-size:.66rem;margin-top:.2rem">'
            f"{'AI vs backtest match' if agreement == 'MATCH' else ('Backtest failed' if agreement == 'BACKTEST_FAILED' else 'Directions differ')}"
            f"</div></div>",
            unsafe_allow_html=True,
        )
    with c5d:
        st.markdown(
            '<div style="background:#0A2540;padding:.6rem 1rem;border-radius:6px;text-align:center">'
            '<div style="color:#8BA9C4;font-size:.7rem;margin-bottom:.3rem">FINAL VERIFIED</div>'
            f'<div style="font-size:1.6rem;font-weight:900">{_decision_badge(final_dec)}</div>'
            f'<div style="color:#8BA9C4;font-size:.62rem;margin-top:.15rem">Options: <b style="color:#FCD34D">{_final_opts_label}</b></div>'
            f'<div style="color:#8BA9C4;font-size:.66rem;margin-top:.1rem">'
            f"{'Confirmed' if agreement == 'MATCH' else 'Human review required'}"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    if agreement == "MATCH":
        st.success(f"MATCH — AI predicted {ai_dec} ({_ai_opts_label}), options backtest P&L direction agrees.")
    elif agreement == "CONFLICT":
        st.error(f"CONFLICT — AI predicted {ai_dec} ({_ai_opts_label}) but backtest result was {_bt_opts_label}.")
    else:
        st.warning(
            f"BACKTEST FAILED — {opts.get('error', opts_status)}. "
            "Agreement cannot be determined. Options accuracy NOT saved."
        )

    if saved:
        st.success(f"Options accuracy record saved. {save_msg}")
    else:
        st.info(f"Record not saved: {save_msg}")

    # Known Answer Audit
    if val_ok:
        _render_known_answer_audit(val, cap)

    # ── Section 9 — Developer Debug ───────────────────────────────────────────
    _section_header(9, "Developer Debug", "Raw JSON, hash proof, backtest payload, no-leakage proof")
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
                ("AI Output Hash",      f'<code style="font-size:.68rem">{ai.get("stock_prediction_input_hash", "---")}</code>'),
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
                ("Backtest = ctx_start?", "NO (correctly uses origin_date)" if origin != ctx_start else "YES (BUG!)"),
                ("Backtest status",     opts_status),
                ("Backtest ID",         opts.get("backtest_id", "N/A")),
                ("Options P&L",         _sign_fmt(opts.get("profit_loss")) if opts_status == "SUCCESS" else "N/A"),
                ("Options win rate",    str(opts.get("win_rate", "N/A")) if opts_status == "SUCCESS" else "N/A"),
                ("Options trades",      str(opts.get("total_trades", "N/A")) if opts_status == "SUCCESS" else "N/A"),
            ]),
            unsafe_allow_html=True,
        )

    with st.expander("StockPredictionInput JSON"):
        st.json(spi)
    with st.expander("Options Params"):
        st.json(opts_p)
    _opts_vtype = opts.get("validation_type", "")
    _bt_payload_title = (
        "Delta Proxy Payload Sent To Tastytrade (APPROXIMATE — not exact strike)"
        if _opts_vtype == "DELTA_PROXY_APPROXIMATE" else
        "Backtest Payload Sent To Tastytrade"
    )
    if opts.get("exact_validation_status") == "UNSUPPORTED_BY_PROVIDER":
        with st.expander("Exact Strike Request (UNSUPPORTED by provider — not sent)"):
            st.json({
                "contract_selection_method": "EXACT_STRIKE",
                "requested_strike":          opts.get("requested_strike"),
                "expiry_date":               opts.get("expiry_date"),
                "exact_validation_status":   "UNSUPPORTED_BY_PROVIDER",
                "eligible_for_exact_accuracy": False,
                "note":                      opts.get("exact_validation_note", "Exact strike backtesting unsupported by Tastytrade API."),
            })
    with st.expander(_bt_payload_title):
        _bt_payload_display = bt_payload if bt_payload else {"note": "Payload not stored (tastytrade unavailable)"}
        if _opts_vtype == "DELTA_PROXY_APPROXIMATE":
            _bt_payload_display = dict(_bt_payload_display) if isinstance(_bt_payload_display, dict) else {}
            _bt_payload_display["_proxy_note"] = (
                "This payload uses delta=50 (ATM proxy) because Tastytrade does not support "
                "fixed-strike backtesting. This is NOT an exact strike result."
            )
        st.json(_bt_payload_display)
    with st.expander("User Input Snapshot vs Effective Input (date binding proof)"):
        _snap_o = st.session_state.get("user_input_snapshot", {})
        _rctx_o = st.session_state.get("run_context", {})
        _ibw_o  = _rctx_o.get("input_binding_warning")
        if _ibw_o:
            st.error(_ibw_o)
        _uo1, _uo2 = st.columns(2)
        with _uo1:
            st.markdown("**User-Entered Inputs (exact, captured at submit)**")
            st.json(_snap_o)
        with _uo2:
            st.markdown("**Effective Inputs Used by AI (after provider constraints)**")
            st.json({
                "effective_ctx_start":    _rctx_o.get("effective_ctx_start") or _snap_o.get("historical_context_start_date", "same as requested"),
                "requested_ctx_start":    _rctx_o.get("requested_ctx_start") or _snap_o.get("historical_context_start_date"),
                "ctx_start_gap_days":     _rctx_o.get("ctx_start_gap_days", 0),
                "provider_status":        _rctx_o.get("provider_status", {}),
                "input_binding_warning":  _ibw_o or "NONE — inputs matched",
            })
    _run_ctx_d = st.session_state.get("run_context", {})
    _tt_dbg_d  = _run_ctx_d.get("tastytrade_debug", {})
    with st.expander("Tastytrade Auth Debug (credential source, auth chain, backtest chain)"):
        _cred_src = _tt_dbg_d.get("credential_source") or (
            "REFRESH_TOKEN" if os.getenv("TASTYTRADE_REFRESH_TOKEN")
            else "ACCESS_TOKEN" if os.getenv("TASTYTRADE_ACCESS_TOKEN")
            else "MISSING"
        )
        st.markdown(
            _table([
                ("Credential Source",     _cred_src),
                ("TT Token Refresh",      _tt_dbg_d.get("token_refresh_status", "NOT_RUN")),
                ("Customer Check",        _tt_dbg_d.get("customer_check_status", "NOT_RUN")),
                ("Backtest Create",       _tt_dbg_d.get("backtest_create", "NOT_RUN")),
                ("Backtest Poll",         _tt_dbg_d.get("backtest_poll", "NOT_RUN")),
                ("Final TT Status",       opts_status),
                ("Accuracy Skip Reason",  opts.get("accuracy_skip_reason", "NONE")),
                ("Accuracy Saved",        str(saved)),
            ]),
            unsafe_allow_html=True,
        )
    with st.expander("Run Context (provider status for this run)"):
        st.json(_run_ctx_d)
    with st.expander("AI Prediction Raw JSON"):
        st.json({k: v for k, v in ai.items() if k != "_momentum_signals"})
    with st.expander("Momentum Signals Used by AI"):
        st.json(mom)
    with st.expander("Options Backtest Result"):
        st.json(opts)
    with st.expander("Actual Stock Validation Result"):
        st.json({k: v for k, v in val.items() if k != "comparison"} if val else {})
    with st.expander("Comparison Raw JSON"):
        st.json(cmp)
    with st.expander("AI Context Bars (first/last 3, no leakage proof)"):
        ctx_bars_all = [b for b in hist if ctx_start <= b["date"] <= origin]
        st.json({
            "n_bars":    len(ctx_bars_all),
            "first_3":   ctx_bars_all[:3],
            "last_3":    ctx_bars_all[-3:],
            "ctx_start": ctx_start,
            "cutoff":    origin,
            "target":    target,
            "note":      "target-date price was NOT in these bars",
        })
    _opts_sig = ai.get("signal_scores", {})
    if _opts_sig:
        with st.expander("Signal Score Engine (bull/bear/uncertainty)"):
            _bull = _opts_sig.get("bullish_score", 0)
            _bear = _opts_sig.get("bearish_score", 0)
            _unc  = _opts_sig.get("uncertainty_score", 0)
            _dom  = _opts_sig.get("dominant", "---")
            _doms = _opts_sig.get("dominant_score", 0)
            st.markdown(
                _table([
                    ("Bullish Score",    f'<b style="color:#10B981">{_bull}/100</b>'),
                    ("Bearish Score",    f'<b style="color:#EF4444">{_bear}/100</b>'),
                    ("Uncertainty Score",f'<b style="color:#F59E0B">{_unc}/100</b>'),
                    ("Dominant Signal",  f'<b>{_dom.upper()} ({_doms}/100)</b>'),
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
        "stock_prediction_evaluation_runs.jsonl -- last 20 runs",
    )
    records = load_stock_prediction_records(limit=50)
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

    import pandas as pd
    rows = []
    for r in records[:20]:
        sp  = r.get("stock_prediction_input") or {}
        ap  = r.get("ai_prediction") or {}
        av  = r.get("actual_validation") or {}
        cmp = r.get("comparison") or {}
        rows.append({
            "Timestamp":       (r.get("timestamp") or "")[:19],
            "Run ID":          r.get("run_id", ""),
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
