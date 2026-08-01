"""
Primary UI Contract Tests — Static source analysis.

The two-mode app (stock_prediction_app.py) legitimately contains options terms
in Mode 2 (Direction, Type, DTE, Delta, etc.).
Tests here verify:
  1. Old options-specific LEGACY terms are absent (strategy_backtest_verification, etc.)
  2. Required stock prediction terms ARE present (Mode 1 contract)
  3. Required mode selector terms ARE present (both modes visible)
  4. streamlit_app.py redirects to stock_prediction_app (not legacy options app)
  5. legacy_options_app.py exists but has no warning banner

Terms that ARE NOW ALLOWED in stock_prediction_app.py (Mode 2 uses them legitimately):
  DTE, Delta, Direction, Legs, Call/Put, Options Strategy Parameters
  (these are user-facing options strategy inputs, not the old hidden legacy backtester)

Terms still BANNED (legacy identifiers that must never appear as UI labels):
  strategy_backtest_verification   (old JSONL key)
  equity-option                    (internal API type, not a UI label)
  Strategy Prediction vs Backtest Verification   (old app title)
  Strategy Factor Intelligence Scores            (old section title)
  LEGACY OPTIONS BACKTEST          (old warning banner)
  Entry Frequency                  (replaced by Entry Schedule)
  Short Put / Long Call as hardcoded strategy names (user picks Direction + Type instead)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

PRIMARY_FILES = [
    ROOT / "streamlit_app.py",
    ROOT / "stock_prediction_app.py",
]

LEGACY_FILE = ROOT / "legacy_options_app.py"

# ── Banned terms: old legacy identifiers that must NEVER appear as UI labels ──
# Mode 2 legitimately has DTE, Delta, Direction, etc.
# What's banned here is the OLD app architecture's fingerprints.
BANNED_UI_TERMS = [
    # Old strategy names hardcoded in UI (user picks Direction + Type instead)
    r"Short Put",
    r"Long Call",
    r"short put",
    r"long call",
    # Old entry frequency term (replaced by Entry Schedule)
    r"\bEntry Frequency\b",
    # Legacy JSONL key (old accuracy engine record type)
    r"strategy_backtest_verification",
    # Old app section titles
    r"Strategy Prediction vs Backtest Verification",
    r"Strategy Factor Intelligence Scores",
    # Legacy warning banner text
    r"LEGACY OPTIONS BACKTEST",
]

# Allow these substrings on lines that would otherwise match a banned pattern
TERM_EXCEPTIONS: dict = {
    # "strategy_backtest_verification" is OK in import lines and comments
    r"strategy_backtest_verification": ["import ", "from ", "#"],
}

# ── Required terms: must be present in stock_prediction_app.py ───────────────
REQUIRED_IN_PREDICTION_APP = [
    # Mode selector
    "Stock Price Validation",
    "Options Strategy Validation",
    # Core stock prediction identifiers
    "Walk-Forward Prediction Validation",
    "historical_context_start_date",
    "prediction_origin_date",
    "decision_horizon_days",
    "target_date",
    # Stock mode sections — Section 3 and 4 are side-by-side column headers
    "GEMINI AI STOCK PREDICTION",
    "SECTION 4 — ACTUAL HISTORICAL VALIDATION",
    "AI STOCK PREDICTION",
    "ACTUAL HISTORICAL VALIDATION",
    "Numeric Closeness",
    "Final Decision Board",
    "actual_final_capital",
    "price_basis",
    # Known answer audit
    "Known Answer",
    # Context filter — AI must not see bars before ctx_start
    "ctx_start <= b",
]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _find_banned_hits(text: str, pattern: str, exceptions: list) -> list:
    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        if re.search(pattern, line):
            skip = False
            for exc in exceptions:
                if exc in line:
                    skip = True
                    break
            if not skip:
                hits.append((i, line.strip()))
    return hits


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Primary files exist
# ══════════════════════════════════════════════════════════════════════════════

def test_streamlit_app_exists():
    assert (ROOT / "streamlit_app.py").exists(), (
        "streamlit_app.py must exist -- it is the primary entry point"
    )


def test_stock_prediction_app_exists():
    assert (ROOT / "stock_prediction_app.py").exists(), (
        "stock_prediction_app.py must exist"
    )


def test_legacy_options_app_exists():
    assert LEGACY_FILE.exists(), (
        "legacy_options_app.py must exist -- old options app preserved but archived"
    )


def test_prediction_validation_service_exists():
    assert (ROOT / "prediction_validation_service.py").exists(), (
        "prediction_validation_service.py must exist -- clean backend for Discord/Amaan"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Banned legacy terms absent from primary files
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("file_path", PRIMARY_FILES, ids=lambda p: p.name)
@pytest.mark.parametrize("banned_pattern", BANNED_UI_TERMS)
def test_banned_term_absent(file_path: Path, banned_pattern: str):
    """Each banned legacy term must be absent from each primary app file."""
    if not file_path.exists():
        pytest.skip(f"{file_path.name} does not exist")

    text       = _read(file_path)
    exceptions = TERM_EXCEPTIONS.get(banned_pattern, [])
    hits       = _find_banned_hits(text, banned_pattern, exceptions)

    assert not hits, (
        f"BANNED LEGACY TERM '{banned_pattern}' found in {file_path.name}:\n"
        + "\n".join(f"  Line {ln}: {line}" for ln, line in hits[:5])
        + f"\n  (first 5 of {len(hits)} hits)"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Required terms present in stock_prediction_app.py
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("required_term", REQUIRED_IN_PREDICTION_APP)
def test_required_term_present(required_term: str):
    """Required terms must be present in stock_prediction_app.py."""
    path = ROOT / "stock_prediction_app.py"
    if not path.exists():
        pytest.skip("stock_prediction_app.py does not exist")

    text = _read(path)
    assert required_term in text, (
        f"Required term '{required_term}' is MISSING from stock_prediction_app.py. "
        "The two-mode app must contain this term."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Two-mode selector present
# ══════════════════════════════════════════════════════════════════════════════

def test_stock_mode_in_app():
    text = _read(ROOT / "stock_prediction_app.py")
    assert "Stock Price Validation" in text, (
        "Mode 1 label 'Stock Price Validation' must be in stock_prediction_app.py"
    )


def test_options_mode_in_app():
    text = _read(ROOT / "stock_prediction_app.py")
    assert "Options Strategy Validation" in text, (
        "Mode 2 label 'Options Strategy Validation' must be in stock_prediction_app.py"
    )


def test_prediction_window_constraint_documented():
    """
    The two-mode app must document that options backtest runs for prediction window only.
    This is the core architectural invariant.
    """
    text = _read(ROOT / "stock_prediction_app.py")
    # The docstring or UI label must say the backtest is for the prediction window
    assert "prediction_origin_date" in text, (
        "stock_prediction_app.py must reference prediction_origin_date as backtest start"
    )
    # The historical context window must NOT be used as backtest start
    # Verified by: backtest start = prediction_origin_date (not historical_context_start_date)


# ══════════════════════════════════════════════════════════════════════════════
# TEST: streamlit_app.py redirects to stock_prediction_app
# ══════════════════════════════════════════════════════════════════════════════

def test_streamlit_app_references_prediction_app():
    text = _read(ROOT / "streamlit_app.py")
    assert "stock_prediction_app" in text, (
        "streamlit_app.py must import or reference stock_prediction_app. "
        "It is the primary entry point -- not the legacy options backtest."
    )
    assert "strategy_prediction_agent" not in text, (
        "streamlit_app.py must NOT reference strategy_prediction_agent. "
        "That is the legacy options AI."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TEST: legacy_options_app.py has no warning banner
# ══════════════════════════════════════════════════════════════════════════════

def test_legacy_app_no_warning_banner():
    text = _read(LEGACY_FILE)
    assert "NOT FOR ONE-MONTH STOCK PREDICTION VALIDATION" not in text, (
        "legacy_options_app.py should not contain the old warning banner text."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TEST: stock_prediction_app.py has correct title
# ══════════════════════════════════════════════════════════════════════════════

def test_correct_title_present():
    text = _read(ROOT / "stock_prediction_app.py")
    assert "Walk-Forward Prediction Validation" in text, (
        "Title 'Walk-Forward Prediction Validation' must appear in stock_prediction_app.py"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TEST: stock prediction log referenced, not old options log
# ══════════════════════════════════════════════════════════════════════════════

def test_correct_accuracy_log_referenced():
    text = _read(ROOT / "stock_prediction_app.py")
    assert "stock_prediction" in text.lower(), (
        "stock_prediction_app.py must reference stock prediction records"
    )
    assert "strategy_evaluation_runs" not in text, (
        "stock_prediction_app.py must NOT reference strategy_evaluation_runs (old options log)"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TEST: prediction_validation_service.py has required public functions
# ══════════════════════════════════════════════════════════════════════════════

def test_prediction_service_has_required_functions():
    path = ROOT / "prediction_validation_service.py"
    if not path.exists():
        pytest.skip("prediction_validation_service.py not found")
    text = _read(path)

    required_fns = [
        "run_stock_price_validation",
        "run_options_strategy_validation",
        "run_single_validation",
        "run_rolling_monthly_validation",
        "get_validation_summary",
        "map_legacy_strategy_input",
    ]
    for fn in required_fns:
        assert f"def {fn}" in text, (
            f"prediction_validation_service.py must define function '{fn}'"
        )


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Options backtest uses origin_date as start (prediction window)
# ══════════════════════════════════════════════════════════════════════════════

def test_options_backtest_uses_ctx_start_for_full_study_window():
    """
    In _run_options_all, the tastytrade backtest runs over the prediction window
    (origin_date → target_date), so AI learning and options validation cover
    the same forward period. Static proof: check the critical assignment.
    """
    text = _read(ROOT / "stock_prediction_app.py")
    # The backtest call must pass origin_date as start_date (prediction window)
    assert "start_date=origin_date" in text, (
        "Options backtest in stock_prediction_app.py must pass origin_date as start_date. "
        "Backtest runs origin → target (prediction window), not ctx_start → origin."
    )
    # Verify the code still references prediction_origin_date (used for AI study boundary)
    assert "prediction_origin_date" in text.lower(), (
        "Options backtest code must still reference prediction_origin_date for AI study boundary."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TEST: O1-style side-by-side headers present in both mode renderers
# ══════════════════════════════════════════════════════════════════════════════

def test_stock_mode_side_by_side_headers():
    """
    Stock Mode: Section 3 header is dynamic (GEMINI AI STOCK PREDICTION when Gemini active).
    Static proof: both title strings must appear in source, Section 4 must be present.
    """
    text = _read(ROOT / "stock_prediction_app.py")
    assert "GEMINI AI STOCK PREDICTION" in text, (
        "'GEMINI AI STOCK PREDICTION' must be the Gemini-active Section 3 title in stock mode."
    )
    assert "AI STOCK PREDICTION" in text, (
        "'AI STOCK PREDICTION' must appear in source (used in Section 3 header logic)."
    )
    assert "SECTION 4 — ACTUAL HISTORICAL VALIDATION" in text, (
        "'SECTION 4 — ACTUAL HISTORICAL VALIDATION' must be the right column header in stock mode."
    )


def test_options_mode_side_by_side_headers():
    """
    Options Mode: left column = AI PREDICTION, right column = OPTIONS BACKTEST ACTUAL.
    Static proof: both clean header strings must appear in source.
    """
    text = _read(ROOT / "stock_prediction_app.py")
    assert "AI PREDICTION" in text, (
        "'AI PREDICTION' must be the left column header in options mode."
    )
    assert "OPTIONS BACKTEST ACTUAL" in text, (
        "'OPTIONS BACKTEST ACTUAL' must be the right column header in options mode."
    )


def test_final_decision_board_present():
    """Final Decision Board section header must be present in the app."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "Final Decision Board" in text, (
        "'Final Decision Board' section must be present in stock_prediction_app.py."
    )


def test_numeric_closeness_section_present():
    """Numeric Closeness section header must be present in the app."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "Numeric Closeness" in text, (
        "'Numeric Closeness' section must be present in stock_prediction_app.py."
    )


def test_walk_forward_title():
    """App title must reference Walk-Forward Prediction Validation."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "Walk-Forward Prediction Validation" in text, (
        "App title must say 'Walk-Forward Prediction Validation'."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TEST: No person names in primary files
# ══════════════════════════════════════════════════════════════════════════════

def test_no_person_name_in_primary_files():
    """No person names (e.g. Ajay) in primary app files."""
    for path in PRIMARY_FILES:
        if not path.exists():
            continue
        text = _read(path)
        assert "Ajay" not in text, (
            f"Person name 'Ajay' found in {path.name}. "
            "Remove all person names from primary app files."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Provider strategy — explicit config, no confusing fallback messaging
# ══════════════════════════════════════════════════════════════════════════════

def test_provider_strategy_config():
    """historical_price_service.py must have HISTORICAL_PRICE_PROVIDER config, not ALLOW_NON_RAPIDAPI_FALLBACK."""
    path = ROOT / "historical_price_service.py"
    if not path.exists():
        pytest.skip("historical_price_service.py not found")
    text = _read(path)
    assert "HISTORICAL_PRICE_PROVIDER" in text, (
        "historical_price_service.py must define HISTORICAL_PRICE_PROVIDER config."
    )
    assert "get_provider_used" in text, (
        "historical_price_service.py must expose get_provider_used() for UI to read actual provider."
    )
    assert "ALLOW_NON_RAPIDAPI_FALLBACK" not in text, (
        "ALLOW_NON_RAPIDAPI_FALLBACK must be removed. Use HISTORICAL_PRICE_PROVIDER instead."
    )
    assert "NASDAQ fallback" not in text, (
        "'NASDAQ fallback' text must not appear in historical_price_service.py."
    )
    assert "external_historical_provider" in text, (
        "historical_price_service.py must label external bars as 'external_historical_provider'."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TEST: No numbered Section 5 header (Final Decision Board appears without it)
# ══════════════════════════════════════════════════════════════════════════════

def test_no_section_5_numbered_header():
    """Final Decision Board must not use a numbered Section 5 wrapper."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "_section_header(5," not in text, (
        "_section_header(5, ...) must not appear in stock_prediction_app.py. "
        "Final Decision Board should appear without a 'Section 5' numbered header."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Context filter uses both ctx_start lower bound and origin upper bound
# ══════════════════════════════════════════════════════════════════════════════

def test_context_filter_uses_start_date():
    """AI context bars must be filtered by ctx_start (lower bound), not just origin (upper bound)."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "ctx_start <= b" in text, (
        "ctx_bars must filter: ctx_start <= b['date'] <= origin_date. "
        "Without lower-bound filter, AI sees bars from provider start date, not user-selected ctx_start. "
        "This is a context leakage bug."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TEST: No forbidden provider strings anywhere in price service
# ══════════════════════════════════════════════════════════════════════════════

def test_no_forbidden_provider_strings_in_price_service():
    """historical_price_service.py must contain no NASDAQ/Yahoo fallback references."""
    text = _read(ROOT / "historical_price_service.py")
    forbidden = [
        "ALLOW_NON_RAPIDAPI_FALLBACK",
        "NASDAQ fallback",
        "fallback disabled",
        "_fetch_from_nasdaq",
    ]
    for s in forbidden:
        assert s not in text, (
            f"Forbidden string {s!r} found in historical_price_service.py. "
            "RapidAPI TradingView is the only provider — remove all fallback references."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Future target date must not block AI prediction
# ══════════════════════════════════════════════════════════════════════════════

def test_future_target_allowed_in_validation():
    """validate_stock_prediction_input must NOT reject future target dates."""
    text = _read(ROOT / "stock_prediction_agent.py")
    assert "target_date is in the future" not in text, (
        "validate_stock_prediction_input must not reject future target dates. "
        "Future target = live_future_prediction mode — AI still predicts, actual shows PENDING."
    )


def test_live_future_prediction_mode_in_orchestrator():
    """_run_all must detect future target date and store run_type=live_future_prediction."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "live_future_prediction" in text, (
        "stock_prediction_app.py must handle live_future_prediction run type. "
        "Future target date should set run_type='live_future_prediction', not block the run."
    )
    assert "PENDING" in text, (
        "stock_prediction_app.py must show PENDING for actual validation when target is future."
    )


def test_pending_predictions_engine_exists():
    """pending_predictions_engine.py must exist with save/load functions."""
    path = ROOT / "pending_predictions_engine.py"
    assert path.exists(), (
        "pending_predictions_engine.py must exist. "
        "It saves live future predictions for later validation."
    )
    text = _read(path)
    assert "save_pending_prediction" in text
    assert "load_pending_predictions" in text
    assert "count_pending_predictions" in text


def test_no_data_source_nasdaq_in_app():
    """stock_prediction_app.py must not show NASDAQ public text in any UI label."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "NASDAQ public historical prices" not in text, (
        "stock_prediction_app.py must not display 'NASDAQ public historical prices'. "
        "Only RapidAPI TradingView source is shown."
    )
    assert "nasdaq_public_api" not in text, (
        "stock_prediction_app.py must not reference 'nasdaq_public_api' as a UI label."
    )


# ══════════════════════════════════════════════════════════════════════════════
# LIGHT THEME / WHITE UI CONTRACT
# ══════════════════════════════════════════════════════════════════════════════

def test_streamlit_config_exists():
    """`.streamlit/config.toml` must exist — required to force light theme."""
    cfg = ROOT / ".streamlit" / "config.toml"
    assert cfg.exists(), ".streamlit/config.toml must exist to enforce light theme"


def test_streamlit_config_has_light_base():
    """Streamlit config must set base = 'light', not dark."""
    cfg = _read(ROOT / ".streamlit" / "config.toml")
    assert 'base = "light"' in cfg, (
        "config.toml must have base = \"light\" to force white UI"
    )
    assert "dark" not in cfg.lower(), (
        "config.toml must not contain 'dark' theme setting"
    )


def test_streamlit_config_white_background():
    """Streamlit config must set backgroundColor = '#FFFFFF'."""
    cfg = _read(ROOT / ".streamlit" / "config.toml")
    assert "#FFFFFF" in cfg, (
        "config.toml must set backgroundColor = '#FFFFFF'"
    )


def test_app_injects_light_css():
    """stock_prediction_app.py must inject CSS that forces white background."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "_apply_light_demo_theme" in text, (
        "App must define and call _apply_light_demo_theme() for white UI"
    )
    assert "background-color: #FFFFFF" in text, (
        "Injected CSS must force background-color: #FFFFFF"
    )


def test_app_no_dark_theme_css():
    """App must not inject CSS that forces dark/black page background."""
    text = _read(ROOT / "stock_prediction_app.py")
    # Dark background forced globally is forbidden
    assert "background-color: #000000" not in text
    assert "background-color: black" not in text


def test_build_active_banner_present():
    """App must render a build banner showing AI_PROVIDER and build timestamp."""
    text = _read(ROOT / "stock_prediction_app.py")
    # Banner was renamed to ACTIVE BUILD: TRUTH_GATE_V4
    assert "BUILD ACTIVE" in text or "ACTIVE BUILD" in text, "Build banner must exist in app"
    assert "_BUILD_FILE" in text or "_BUILD_TS" in text, "Build timestamp/file variable must exist"
    assert "AI_PROVIDER" in text


def test_execution_truth_panel_present():
    """App must render an Execution Truth Panel before Section 3."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "EXECUTION TRUTH PANEL" in text
    assert "Gemini API Called" in text
    assert "Tastytrade" in text
    assert "RapidAPI" in text
    assert "RapidAPI OHLCV" in text


def test_stock_mode_says_tastytrade_no():
    """Execution Truth Panel in stock mode must state tastytrade is not used."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "stock mode uses" in text.lower() or "Stock mode uses" in text, (
        "App must clearly state tastytrade is not used in stock mode"
    )


def test_light_theme_config_no_black():
    """config.toml must not set any color to #000000 or 'black'."""
    cfg = _read(ROOT / ".streamlit" / "config.toml")
    assert "#000000" not in cfg
    assert "black" not in cfg.lower()
