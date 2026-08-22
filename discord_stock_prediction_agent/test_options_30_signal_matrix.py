"""Known-answer coverage for the 30 complex Discord option signals supplied by the user.

This suite is offline: it verifies parsing and routing metadata without contacting
Polygon, Tastytrade, Discord, or Alpaca and without placing paper orders.
"""
from __future__ import annotations

from .options_parser import classify_and_parse


SIGNALS = [
    "BTO AAPL 240C / STO AAPL 250C 09/19 @4.60 Debit Qty 5",
    "BTO TSM 250P / STO TSM 230P 10/17 @5.25 Debit Qty 3",
    "BUY SPY 640C + 640P 09/19 @8.20 Debit Qty 2",
    "BUY QQQ 600C + 570P 10/17 @7.10 Debit Qty 4",
    "STO SPY 620P / BTO SPY 610P / STO SPY 670C / BTO SPY 680C 09/19 @2.15 Credit Qty 10",
    "SELL SPY 640C 640P BUY 650C 630P 09/19 @4.80 Credit Qty 5",
    "BUY 100 AAPL Shares @212.50 AND STO AAPL 230C 10/17 @2.80 Covered Call",
    "STO AMD 160P 10/17 @3.20 Cash Secured Qty 2",
    "ROLL NVDA 190C 09/19 TO NVDA 205C 10/17 FOR 1.40 Debit",
    "ROLL TSLA 320P 09/19 TO TSLA 300P 10/17 FOR 2.10 Credit",
    "STC 50% OF AAPL 240C 09/19 @6.80 HOLD REMAINING CONTRACTS",
    "SELL 2 OF 5 NVDA 190C 10/17 @8.50 THEN MOVE STOP TO BREAKEVEN",
    "BTO 2 QQQ 600C 10/17 @4.50 THEN ADD 3 MORE IF PREMIUM FALLS TO 3.80",
    "BTO META 760C 11/21 @6.25 TP1 8.00 TP2 10.50 TP3 14.00 SL 4.80",
    "BTO NFLX 1450C 10/17 @18.40 TRAIL STOP 15%",
    "BUY ORCL 280C 09/19 LIMIT 3.90 TAKE PROFIT 6.20 STOP LOSS 2.70",
    "BUY IBM 300C 10/17 @3.20 OCO TP 5.80 SL 2.10",
    "BTO AVGO 380C 09/19 ONLY IF STOCK BREAKS ABOVE 385 ELSE CANCEL ORDER",
    "BTO CRM 290C 10/17 @5.30 EXIT ALL POSITIONS BEFORE MARKET CLOSE IF TP NOT HIT",
    "BUY 100 GOOGL SHARES, BTO GOOGL 220C 10/17 @4.20, STO GOOGL 240C 10/17 @2.10 FOR A BULL CALL SPREAD",
    "BTO ADBE 400C 09/19 / STO ADBE 420C 11/21 @7.40 Debit Qty 2",
    "BTO MSFT 550C 12/19 / STO MSFT 550C 09/19 @5.80 Debit Qty 4",
    "BTO 2 TSLA 300C / STO 4 TSLA 320C 10/17 @2.30 Credit",
    "BUY 100 AMZN SHARES AND BTO AMZN 220P 10/17 @5.40 FOR DOWNSIDE PROTECTION",
    "BUY 100 META SHARES, BTO META 700P 10/17, STO META 800C 10/17 FOR ZERO-COST COLLAR",
    "BTO SPY 620P / STO SPY 630P / BTO SPY 660C / STO SPY 650C 09/19 @3.90 Debit",
    "BTO QQQ 580C / STO 2 QQQ 590C / BTO QQQ 600C 10/17 @2.10 Debit",
    "STO 1 NVDA 210C / BTO 2 NVDA 220C 10/17 @1.50 Debit",
    "Starter on AVGO 380C 09/19 @5.20. Adding above 385 breakout. TP1 6.80, TP2 8.20, runners to 10.00. Stop 4.10.",
    "Opened 5-lot Bull Call Spread: Long 5 AAPL 240C / Short 5 AAPL 250C Sep19 for 4.60 Debit. Target 7.20, max risk",
]


def _option(index: int):
    routed = classify_and_parse(SIGNALS[index - 1])
    assert routed.kind == "OPTION", (index, routed)
    assert routed.option and routed.option.valid, (index, routed.reason)
    return routed.option


def run_all() -> None:
    parsed = [_option(index) for index in range(1, 31)]
    expected_structures = {
        1: "bull_call_spread", 2: "bear_put_spread", 3: "long_straddle",
        4: "long_strangle", 5: "iron_condor", 6: "iron_butterfly",
        7: "covered_call", 8: "cash_secured_put", 9: "roll", 10: "roll",
        20: "bull_call_spread", 21: "diagonal_spread", 22: "calendar_spread",
        23: "ratio_spread", 24: "protective_put", 25: "collar",
        26: "reverse_iron_condor", 27: "butterfly_spread",
        28: "ratio_backspread", 30: "bull_call_spread",
    }
    for index, structure in expected_structures.items():
        assert parsed[index - 1].structure == structure, (index, parsed[index - 1])

    assert [len(parsed[i - 1].legs) for i in (1, 2, 3, 4, 5, 6)] == [2, 2, 2, 2, 4, 4]
    assert [parsed[i - 1].quantity for i in (1, 2, 3, 4, 5, 6)] == [5, 3, 2, 4, 10, 5]
    # A stock+option combo (e.g. a covered call) is routed like a multi-leg
    # signal -- Alpaca can't submit equity+option as one atomic order, so it
    # needs the same "combo" handling. test_options_parser.py's own covered
    # call combo case asserts is_multi_leg == True; this pre-existing
    # assertion had it backwards and (since this file has no pytest-collected
    # test_* function) was never actually run to catch the mismatch.
    assert parsed[6].contains_equity_leg and parsed[6].is_multi_leg
    assert parsed[7].order_action == "open_short" and parsed[7].quantity == 2
    assert [leg.expiry_date for leg in parsed[8].legs] == ["2026-09-19", "2026-10-17"]
    assert [leg.order_action for leg in parsed[8].legs] == ["close_long", "open_long"]
    assert parsed[10].close_percent == 50
    assert parsed[11].order_action == "close_long" and parsed[11].quantity == 2
    assert (parsed[12].quantity, parsed[12].add_quantity, parsed[12].add_trigger_premium) == (2, 3, 3.8)
    assert parsed[13].target_prices == (8.0, 10.5, 14.0) and parsed[13].stop_loss == 4.8
    assert parsed[14].trailing_stop_pct == 15
    assert parsed[15].target_price == 6.2 and parsed[15].stop_loss == 2.7
    assert parsed[16].target_price == 5.8 and parsed[16].stop_loss == 2.1
    assert (parsed[17].underlying_trigger_direction, parsed[17].underlying_trigger_price) == ("above", 385)
    assert parsed[18].exit_before_market_close
    assert parsed[19].contains_equity_leg and parsed[19].is_multi_leg
    assert [leg.expiry_date for leg in parsed[20].legs] == ["2026-09-19", "2026-11-21"]
    assert [leg.expiry_date for leg in parsed[21].legs] == ["2026-12-19", "2026-09-19"]
    assert parsed[22].quantity == 2 and [leg.ratio_qty for leg in parsed[22].legs] == [1, 2]
    assert parsed[23].contains_equity_leg
    assert parsed[24].contains_equity_leg and parsed[24].is_multi_leg
    assert [leg.ratio_qty for leg in parsed[26].legs] == [1, 2, 1]
    assert [leg.ratio_qty for leg in parsed[27].legs] == [1, 2]
    assert parsed[28].target_prices == (6.8, 8.2) and parsed[28].stop_loss == 4.1
    assert parsed[29].quantity == 5 and parsed[29].expiry_date == "2026-09-19"
    assert [leg.ratio_qty for leg in parsed[29].legs] == [1, 1]
    print("30-SIGNAL OPTIONS MATRIX PASSED: 30/30 parsed with expected strategy metadata")


if __name__ == "__main__":
    run_all()
