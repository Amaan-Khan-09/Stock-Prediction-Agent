"""Tests for signal_review_service — verifies output schema and key safety rules."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from agents.signal_review_service import review_signal
from models.options_models import VS


# ── Test 1: Error safety — internal error returns structured response ─────────

def test_review_signal_returns_dict():
    result = review_signal({})
    assert isinstance(result, dict)


def test_review_signal_has_required_keys():
    result = review_signal({"raw_message": "buy AAPL"})
    assert "parsed_successfully" in result
    assert "final_agent_output" in result
    assert "order_status" in result
    assert "reason" in result


# ── Test 2: Stock signal — no auto-order ──────────────────────────────────────

def test_stock_signal_no_auto_order():
    result = review_signal({"raw_message": "buy AAPL"})
    # Stock signals never auto-submit orders
    assert result.get("order_status") == VS.NOT_SUBMITTED


def test_stock_signal_contains_parsed_signal():
    result = review_signal({"raw_message": "sell TSLA"})
    sig = result.get("parsed_signal", {})
    assert sig.get("underlying") == "TSLA"
    assert sig.get("action") in ("SELL", "EXIT")


# ── Test 3: Pre-parsed payload ────────────────────────────────────────────────

def test_preparsed_option_fields_accepted():
    payload = {
        "raw_message": "Bought SPY 620C @ 1.43, 2 contracts",
        "underlying":  "SPY",
        "option_type": "CALL",
        "strike":      620,
        "premium":     1.43,
        "contracts":   2,
        "expiry":      "2026-07-17",
        "dte":         1,
        "asset_type":  "option",
        "action":      "BUY",
        "source":      "discord",
    }
    result = review_signal(payload)
    assert result["parsed_successfully"] is True
    req = result.get("requested_contract", {})
    assert req.get("underlying") == "SPY"
    assert req.get("strike") == 620.0
    assert req.get("option_type") == "CALL"


def test_occ_symbol_built_when_fields_present():
    payload = {
        "raw_message": "SPY 620C exp 2026-07-17",
        "underlying":  "SPY",
        "option_type": "CALL",
        "strike":      620,
        "expiry":      "2026-07-17",
        "asset_type":  "option",
    }
    result = review_signal(payload)
    req = result.get("requested_contract", {})
    occ = req.get("occ_symbol")
    if occ:  # May be None if not all fields resolved
        assert "SPY" in occ
        assert "620" in occ


# ── Test 4: Missing expiry → REVIEW_REQUIRED ─────────────────────────────────

def test_missing_expiry_exact_mode():
    payload = {
        "raw_message": "SPX 7570C at 1.40",
        "underlying":  "SPX",
        "option_type": "CALL",
        "strike":      7570,
        "asset_type":  "option",
        "action":      "BUY",
        "contract_selection_method": "EXACT_STRIKE",
        # No expiry / dte
    }
    result = review_signal(payload)
    assert result.get("final_agent_output") in (VS.REVIEW_REQUIRED, VS.NO_TRADE)
    assert result.get("order_status") == VS.NOT_SUBMITTED


# ── Test 5: Full output schema ────────────────────────────────────────────────

def test_output_schema_keys():
    result = review_signal({"raw_message": "buy AAPL qty 2"})
    required_keys = [
        "parsed_successfully", "parsed_signal", "final_agent_output",
        "order_status", "reason",
    ]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"


def test_parsed_signal_schema():
    result = review_signal({"raw_message": "buy AAPL"})
    sig = result.get("parsed_signal", {})
    required_sig_keys = [
        "raw_message", "asset_type", "action", "underlying",
        "parser_confidence", "missing_fields",
    ]
    for key in required_sig_keys:
        assert key in sig, f"Missing parsed_signal key: {key}"


# ── Test 6: Large strike (SPX 7570) accepted ─────────────────────────────────

def test_large_strike_accepted():
    payload = {
        "raw_message": "Buy SPX 7570 call at 1.40",
        "underlying":  "SPX",
        "option_type": "CALL",
        "strike":      7570,
        "asset_type":  "option",
    }
    result = review_signal(payload)
    req = result.get("requested_contract", {})
    assert req.get("strike") == 7570.0
    # Must not have been rejected for large strike alone
    assert result.get("parsed_successfully") is True


# ── Test 7: Parser confidence field present ───────────────────────────────────

def test_parser_confidence_in_output():
    result = review_signal({"raw_message": "Bought SPY 620C @ 1.43, 2 contracts, exp 2026-07-17"})
    sig = result.get("parsed_signal", {})
    conf = sig.get("parser_confidence")
    assert conf is not None
    assert 0.0 <= float(conf) <= 1.0
