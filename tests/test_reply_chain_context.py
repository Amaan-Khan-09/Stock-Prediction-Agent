"""Tests for same-day reply chain context memory."""
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from agents.discord_parser import parse_signal
from agents.context_memory import (
    record_signal, get_all_trades, get_open_trades,
    summarize_day, _load_day, _save_day,
)
from models.options_models import SignalStatus, AssetType, OptionSide


# Use a fake test date to avoid polluting real day files
TEST_DATE = "2099-01-15"


def _clear_test_day():
    """Remove the test-day file if it exists."""
    from agents.context_memory import _day_file
    f = _day_file(TEST_DATE)
    if f.exists():
        f.unlink()


# ── Test 1: Entry creates OPEN trade ─────────────────────────────────────────

def test_entry_creates_open_trade():
    _clear_test_day()
    entry_msg_id = str(uuid.uuid4())[:8]
    sig = parse_signal("Entered SPY 620C 0DTE @ 1.40, 2 contracts")
    sig.expiry_date = "2099-01-15"  # force to test date
    trade = record_signal(sig, message_id=entry_msg_id, day=TEST_DATE)
    assert trade.status == SignalStatus.OPEN
    assert trade.contracts_opened == 2
    assert trade.remaining_contracts == 2
    assert trade.entry_price == 1.40
    assert trade.underlying == "SPY"


# ── Test 2: Full reply chain test — entry + trim + exit ──────────────────────

def test_reply_chain_entry_trim_exit():
    """The core test from the spec: $130 P&L from 2 contracts."""
    _clear_test_day()

    # Entry message
    entry_id = str(uuid.uuid4())[:8]
    entry_sig = parse_signal("Entered SPY 620C 0DTE @ 1.40, 2 contracts")
    entry_sig.expiry_date = "2099-01-15"
    entry_trade = record_signal(entry_sig, message_id=entry_id, day=TEST_DATE)

    assert entry_trade.status == SignalStatus.OPEN
    assert entry_trade.contracts_opened == 2

    # Reply 1: trim half at 1.90
    trim_sig = parse_signal("trimmed half at 1.90, runners left")
    trim_sig.reply_to_message_id = entry_id
    trim_sig.underlying = "SPY"  # set so we can find the linked trade
    trim_trade = record_signal(trim_sig, day=TEST_DATE)

    # Should be same trade (PARTIAL)
    assert trim_trade.trade_id == entry_trade.trade_id
    assert trim_trade.status == SignalStatus.PARTIAL
    assert trim_trade.contracts_closed == 1
    assert trim_trade.remaining_contracts == 1

    # Reply 2: out rest at 2.20
    exit_sig = parse_signal("out rest at 2.20")
    exit_sig.reply_to_message_id = entry_id
    exit_sig.underlying = "SPY"
    exit_trade = record_signal(exit_sig, day=TEST_DATE)

    # Should be same trade (CLOSED)
    assert exit_trade.trade_id == entry_trade.trade_id
    assert exit_trade.status == SignalStatus.CLOSED
    assert exit_trade.remaining_contracts == 0

    # P&L check: (1.90 - 1.40)*1*100 + (2.20 - 1.40)*1*100 = 50 + 80 = 130
    pnl = exit_trade.realized_pnl
    assert pnl is not None, "P&L should be computed"
    assert abs(pnl - 130.0) < 0.01, f"Expected $130.00 P&L, got ${pnl:.2f}"


# ── Test 3: Two separate trades on same day ───────────────────────────────────

def test_two_separate_trades():
    _clear_test_day()

    # Trade 1: SPY
    sig1 = parse_signal("Entered SPY 620C 0DTE @ 1.40, 2 contracts")
    sig1.expiry_date = "2099-01-15"
    trade1 = record_signal(sig1, message_id="MSG1", day=TEST_DATE)

    # Trade 2: AAPL (different underlying)
    sig2 = parse_signal("Bought AAPL 240C @ 0.82, 1 contract, exp 2026-08-21")
    trade2 = record_signal(sig2, message_id="MSG2", day=TEST_DATE)

    assert trade1.trade_id != trade2.trade_id
    assert len(get_all_trades(TEST_DATE)) == 2


# ── Test 4: Day summary ────────────────────────────────────────────────────────

def test_day_summary_after_closed_trade():
    _clear_test_day()

    entry_id = "SUMENTRY"
    sig = parse_signal("Entered SPY 620C 0DTE @ 1.40, 2 contracts")
    sig.expiry_date = "2099-01-15"
    record_signal(sig, message_id=entry_id, day=TEST_DATE)

    trim = parse_signal("trimmed half at 1.90")
    trim.reply_to_message_id = entry_id
    trim.underlying = "SPY"
    record_signal(trim, day=TEST_DATE)

    exit_ = parse_signal("out rest at 2.20")
    exit_.reply_to_message_id = entry_id
    exit_.underlying = "SPY"
    record_signal(exit_, day=TEST_DATE)

    summary = summarize_day(TEST_DATE)
    assert summary["closed"] == 1
    assert summary["open"]   == 0
    assert abs(summary["total_realized_pnl"] - 130.0) < 0.01


# ── Test 5: Open trades query ─────────────────────────────────────────────────

def test_open_trades_query():
    _clear_test_day()
    sig = parse_signal("Entered SPY 620C 0DTE @ 1.40, 2 contracts")
    sig.expiry_date = "2099-01-15"
    record_signal(sig, message_id="OPEN1", day=TEST_DATE)

    open_trades = get_open_trades(TEST_DATE)
    assert len(open_trades) == 1
    assert open_trades[0].status == SignalStatus.OPEN
