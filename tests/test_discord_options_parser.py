"""Tests for Discord/WhatsApp natural-language options and stock signal parser."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from agents.discord_parser import parse_signal
from models.options_models import AssetType, ContractMethod, OptionSide, SignalAction


# ── Test 1: Exact strike parse ────────────────────────────────────────────────

def test_parse_aapl_240c_exact():
    sig = parse_signal("Bought AAPL 240C @ 0.82, 1 contract, exp 2026-08-21")
    assert sig.underlying == "AAPL"
    assert sig.strike == 240.0
    assert sig.option_type == OptionSide.CALL
    assert sig.premium == 0.82
    assert sig.contracts == 1
    assert sig.expiry_date == "2026-08-21"
    assert sig.contract_selection_method == ContractMethod.EXACT_STRIKE
    assert sig.asset_type == AssetType.OPTION


def test_parse_spy_620c_entry():
    sig = parse_signal("Entered SPY 620C 0DTE @ 1.40, 2 contracts")
    assert sig.underlying == "SPY"
    assert sig.strike == 620.0
    assert sig.option_type == OptionSide.CALL
    assert sig.premium == 1.40
    assert sig.contracts == 2
    assert sig.dte == 0
    assert sig.asset_type == AssetType.OPTION


def test_parse_spy_620c_with_expiry():
    sig = parse_signal("Bought SPY 620C @ 1.43, 2 contracts, exp 2026-07-17")
    assert sig.underlying == "SPY"
    assert sig.strike == 620.0
    assert sig.option_type == OptionSide.CALL
    assert sig.premium == 1.43
    assert sig.contracts == 2
    assert sig.expiry_date == "2026-07-17"


# ── Test 2: Delta selection parse ─────────────────────────────────────────────

def test_parse_googl_call_delta():
    sig = parse_signal("Buy GOOGL call delta 83 DTE 45 qty 1")
    assert sig.asset_type == AssetType.OPTION
    assert sig.underlying == "GOOGL"
    assert sig.option_type == OptionSide.CALL
    assert sig.delta_ui == 83
    assert abs(sig.delta_decimal - 0.83) < 0.001
    assert sig.dte == 45
    assert sig.contract_selection_method == ContractMethod.DELTA_SELECTION


def test_delta_normalization():
    sig = parse_signal("Buy SPY call delta 30 DTE 45")
    assert sig.delta_ui == 30
    assert abs(sig.delta_decimal - 0.30) < 0.001


# ── Test 3: Large strike (SPX 7570) ──────────────────────────────────────────

def test_parse_spx_7570_call():
    sig = parse_signal("Buy SPX 7570 call at 1.40")
    assert sig.underlying == "SPX"
    assert sig.strike == 7570.0, f"Got strike={sig.strike} — large strikes must NOT be capped"
    assert sig.option_type == OptionSide.CALL
    assert sig.premium == 1.40


def test_parse_spx_inline_7570c():
    sig = parse_signal("SPX7570C lotto @ 1.40")
    assert sig.strike == 7570.0
    assert sig.option_type == OptionSide.CALL


# ── Test 4: Missing expiry → missing_fields ───────────────────────────────────

def test_missing_expiry_flags():
    sig = parse_signal("SPX 7570C at 1.40")
    assert sig.asset_type == AssetType.OPTION
    assert sig.strike == 7570.0
    assert "expiry_or_dte" in sig.missing_fields


def test_options_with_dte_no_expiry():
    sig = parse_signal("Buy SPY 450P, 0DTE")
    assert sig.dte == 0
    assert "expiry_or_dte" not in sig.missing_fields  # DTE is present


# ── Test 5: Stock parse ───────────────────────────────────────────────────────

def test_parse_stock_buy():
    sig = parse_signal("buy AAPL")
    assert sig.asset_type == AssetType.STOCK
    assert sig.underlying == "AAPL"
    assert sig.action == SignalAction.BUY


def test_parse_stock_with_qty():
    sig = parse_signal("buy MSFT qty 2")
    assert sig.asset_type == AssetType.STOCK
    assert sig.underlying == "MSFT"
    assert sig.quantity == 2


def test_parse_stock_sell():
    sig = parse_signal("sell TSLA")
    assert sig.underlying == "TSLA"
    assert sig.action == SignalAction.SELL


def test_parse_stock_hold():
    sig = parse_signal("holding NVDA")
    assert sig.underlying == "NVDA"
    assert sig.action == SignalAction.HOLD


def test_parse_stock_exit():
    sig = parse_signal("exit MSFT")
    assert sig.underlying == "MSFT"
    assert sig.action in (SignalAction.SELL, SignalAction.EXIT)


def test_parse_stock_trim():
    sig = parse_signal("trim half of AMD")
    assert sig.underlying == "AMD"
    assert sig.action == SignalAction.TRIM


# ── Test 6: Partial exit / reply patterns ────────────────────────────────────

def test_parse_trimmed_at_price():
    sig = parse_signal("trimmed half at 1.90, runners left")
    assert sig.action == SignalAction.TRIM
    assert sig.premium == 1.90


def test_parse_out_rest():
    sig = parse_signal("out rest at 2.20")
    assert sig.action in (SignalAction.SELL, SignalAction.EXIT)
    assert sig.premium == 2.20


def test_parse_sold_off_this_one():
    sig = parse_signal("Sold off this one")
    assert sig.action == SignalAction.SELL


def test_parse_holding_runners():
    sig = parse_signal("Holding runners")
    assert sig.action == SignalAction.HOLD


# ── Test 7: Put option ────────────────────────────────────────────────────────

def test_parse_spy_450p_lotto():
    sig = parse_signal("SPY 450P lotto")
    assert sig.underlying == "SPY"
    assert sig.strike == 450.0
    assert sig.option_type == OptionSide.PUT
    assert sig.high_risk_tag is True


def test_parse_tsla_300_puts_weekly():
    sig = parse_signal("TSLA 300 puts weekly")
    assert sig.underlying == "TSLA"
    assert sig.strike == 300.0
    assert sig.option_type == OptionSide.PUT


# ── Test 8: Natural language company names ────────────────────────────────────

def test_parse_apple_company_name():
    sig = parse_signal("Apple looks strong here, taking small buy")
    assert sig.underlying == "AAPL"
    assert sig.action == SignalAction.BUY
