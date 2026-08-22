"""Known-answer tests for options_parser.py and options_symbol.py.

Pure functions only -- no network calls, no Discord, no Alpaca. Same style
as stock_validation_test_harness.py in the project root.

Run (from the project root, package-relative imports require -m):
    venv\\Scripts\\python.exe -m discord_stock_prediction_agent.test_options_parser
"""
from __future__ import annotations

from datetime import date

from .options_parser import classify_and_parse, parse_option_signal
from .options_symbol import (
    build_occ_symbol,
    default_expiry_date,
    parse_occ_symbol,
    resolve_underlying_for_prediction,
)
from .signal_parser import parse_signal

PASS = 0
FAIL = 0


def _assert(condition: bool, name: str, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}" + (f" -- {detail}" if detail else ""))


def test_occ_symbol_known_answer() -> None:
    print("\nTest: OCC symbol construction (known answer)")
    sym = build_occ_symbol("SPX", date(2026, 7, 16), "CALL", 7570.0)
    _assert(sym == "SPX260716C07570000", "SPX 2026-07-16 7570 CALL", f"got {sym}")

    sym2 = build_occ_symbol("AAPL", date(2024, 1, 19), "PUT", 150.5)
    _assert(sym2 == "AAPL240119P00150500", "AAPL fractional strike PUT", f"got {sym2}")

    parsed_back = parse_occ_symbol(sym)
    _assert(parsed_back is not None and parsed_back["root"] == "SPX", "parse_occ_symbol root roundtrip")
    _assert(parsed_back is not None and parsed_back["strike"] == 7570.0, "parse_occ_symbol strike roundtrip")
    _assert(parsed_back is not None and parsed_back["side"] == "CALL", "parse_occ_symbol side roundtrip")


def test_index_proxy_mapping() -> None:
    print("\nTest: index-to-ETF proxy mapping for prediction")
    _assert(resolve_underlying_for_prediction("SPX") == "SPY", "SPX -> SPY")
    _assert(resolve_underlying_for_prediction("NDX") == "QQQ", "NDX -> QQQ")
    _assert(resolve_underlying_for_prediction("AAPL") == "AAPL", "AAPL passes through unchanged")


def test_user_example_journal_single_leg() -> None:
    print("\nTest: the exact reported failing signal now parses as an OPTION journal entry")
    text = "Bought SPX 7570C at 1.40 EOD Lotto. Strangle is complete."
    routed = classify_and_parse(text)
    _assert(routed.kind == "OPTION", "kind == OPTION", f"got {routed.kind}")
    opt = routed.option
    _assert(opt is not None and opt.valid, "option parsed as valid")
    _assert(opt.root == "SPX", "root == SPX", f"got {opt.root}")
    _assert(opt.strike == 7570.0, "strike == 7570.0", f"got {opt.strike}")
    _assert(opt.side == "CALL", "side == CALL", f"got {opt.side}")
    _assert(opt.fill_price == 1.40, "fill_price == 1.40", f"got {opt.fill_price}")
    _assert(opt.tense == "past", "tense == past (Bought)", f"got {opt.tense}")
    _assert(opt.structure == "strangle", "structure == strangle", f"got {opt.structure}")


def test_present_tense_new_order() -> None:
    print("\nTest: present-tense imperative options signal routes to new_order")
    text = "Buy SPX 7570C at 1.40"
    opt = parse_option_signal(text)
    _assert(opt.valid, "option valid")
    _assert(opt.tense == "new_order", "tense == new_order (Buy)", f"got {opt.tense}")
    _assert(opt.side == "CALL", "side == CALL")


def test_natural_language_option_order() -> None:
    print("\nTest: natural-language options signals parse as actionable options")
    routed = classify_and_parse(
        "We need to buy call option of SPY for strike rate 7570 expires date same day"
    )
    _assert(routed.kind == "OPTION", "natural-language SPY call -> OPTION", f"got {routed.kind}")
    opt = routed.option
    _assert(opt is not None and opt.valid, "natural-language option valid")
    _assert(opt is not None and opt.root == "SPY", "root == SPY", f"got {getattr(opt, 'root', None)}")
    _assert(opt is not None and opt.strike == 7570.0, "strike == 7570.0", f"got {getattr(opt, 'strike', None)}")
    _assert(opt is not None and opt.side == "CALL", "side == CALL", f"got {getattr(opt, 'side', None)}")
    _assert(opt is not None and opt.expiry_mode == "0dte", "same day -> 0dte", f"got {getattr(opt, 'expiry_mode', None)}")
    _assert(opt is not None and opt.tense == "new_order", "tense == new_order", f"got {getattr(opt, 'tense', None)}")


def test_ce_pe_and_premium_formats() -> None:
    print("\nTest: CE/PE and premium/@ formats parse correctly")
    ce = classify_and_parse("Buy AAPL 220 CE premium 6.40").option
    _assert(ce is not None and ce.valid, "AAPL CE valid")
    _assert(ce is not None and ce.side == "CALL", "CE -> CALL", f"got {getattr(ce, 'side', None)}")
    _assert(ce is not None and ce.fill_price == 6.40, "premium 6.40 parsed", f"got {getattr(ce, 'fill_price', None)}")

    pe = classify_and_parse("Buy TSLA 300 PE @ 4.25 qty 2").option
    _assert(pe is not None and pe.valid, "TSLA PE valid")
    _assert(pe is not None and pe.side == "PUT", "PE -> PUT", f"got {getattr(pe, 'side', None)}")
    _assert(pe is not None and pe.fill_price == 4.25, "@ 4.25 parsed", f"got {getattr(pe, 'fill_price', None)}")
    _assert(pe is not None and pe.quantity == 2.0, "qty 2 parsed", f"got {getattr(pe, 'quantity', None)}")


def test_delta_option_signal() -> None:
    print("\nTest: explicit delta options signals parse without strike")
    routed = classify_and_parse("Buy 30 delta AAPL call qty 1")
    _assert(routed.kind == "OPTION", "delta call -> OPTION", f"got {routed.kind}")
    opt = routed.option
    _assert(opt is not None and opt.valid, "delta option valid", getattr(opt, "reason", ""))
    _assert(opt is not None and opt.root == "AAPL", "root == AAPL", f"got {getattr(opt, 'root', None)}")
    _assert(opt is not None and opt.side == "CALL", "side == CALL", f"got {getattr(opt, 'side', None)}")
    _assert(opt is not None and opt.strike is None, "no strike for delta signal", f"got {getattr(opt, 'strike', None)}")
    _assert(opt is not None and opt.delta_target == 30.0, "delta target == 30", f"got {getattr(opt, 'delta_target', None)}")

    put = classify_and_parse("BTO 20 delta TSLA put").option
    _assert(put is not None and put.valid, "delta put valid", getattr(put, "reason", ""))
    _assert(put is not None and put.root == "TSLA", "put root == TSLA", f"got {getattr(put, 'root', None)}")
    _assert(put is not None and put.side == "PUT", "put side == PUT", f"got {getattr(put, 'side', None)}")
    _assert(put is not None and put.delta_target == 20.0, "put delta target == 20", f"got {getattr(put, 'delta_target', None)}")


def test_mixed_options_routing_matrix() -> None:
    print("\nTest: mixed options routing matrix")
    cases = [
        ("BTO AAPL 240C 08/21 @3.45", "OPTION", True, "AAPL", 240.0, "CALL", None),
        ("BUY SPY 600 CE qty 1", "OPTION", True, "SPY", 600.0, "CALL", None),
        ("BUY TSLA 300P 07/24 MARKET", "OPTION", True, "TSLA", 300.0, "PUT", None),
        ("Buy 30 delta AAPL call", "OPTION", True, "AAPL", None, "CALL", 30.0),
        ("BTO 20 delta TSLA put", "OPTION", True, "TSLA", None, "PUT", 20.0),
        ("Long 45 delta NVDA calls qty 2", "OPTION", True, "NVDA", None, "CALL", 45.0),
        ("SELL SPY 615/610/665/670 IC 08/21 @2.10 Credit", "OPTION", True, "SPY", None, None, None),
        ("Move stop to breakeven.", "OPTION", False, "", None, None, None),
        ("No clear edge today.", "NO_TRADE", False, "", None, None, None),
    ]
    for text, kind, valid, root, strike, side, delta in cases:
        routed = classify_and_parse(text)
        _assert(routed.kind == kind, f"{text} -> {kind}", f"got {routed.kind}")
        if routed.kind != "OPTION":
            continue
        opt = routed.option
        _assert(opt is not None and opt.valid == valid, f"{text} valid == {valid}", getattr(opt, "reason", ""))
        if root:
            _assert(opt is not None and opt.root == root, f"{text} root == {root}", f"got {getattr(opt, 'root', None)}")
        _assert(opt is not None and opt.strike == strike, f"{text} strike == {strike}", f"got {getattr(opt, 'strike', None)}")
        _assert(opt is not None and opt.side == side, f"{text} side == {side}", f"got {getattr(opt, 'side', None)}")
        _assert(opt is not None and opt.delta_target == delta, f"{text} delta == {delta}", f"got {getattr(opt, 'delta_target', None)}")



def test_real_world_discord_option_variants() -> None:
    print("\nTest: real-world Discord option variants parse correctly")
    today = date.today()
    july_24 = date(today.year, 7, 24)
    next_july_24 = july_24 if july_24 >= today else date(today.year + 1, 7, 24)
    default_expiry = default_expiry_date("0dte").isoformat()
    cases = [
        ("AAPL $240 calls 8/21 @ 3.45", "AAPL", 240.0, "CALL", "2026-08-21", 1.0, 3.45, None, None),
        ("AAPL 8/21 240C @3.45", "AAPL", 240.0, "CALL", "2026-08-21", 1.0, 3.45, None, None),
        ("8/21 AAPL 240C @3.45", "AAPL", 240.0, "CALL", "2026-08-21", 1.0, 3.45, None, None),
        ("calls on Apple strike 240 exp Aug 21", "AAPL", 240.0, "CALL", "2026-08-21", 1.0, None, None, None),
        ("Buy Microsoft 520C Sep 18 @7.10", "MSFT", 520.0, "CALL", "2026-09-18", 1.0, 7.10, None, None),
        ("BTO TSLA Jul 24 290P @5.60", "TSLA", 290.0, "PUT", next_july_24.isoformat(), 1.0, 5.60, None, None),
        ("AAPL 240C 21 Aug @3.45", "AAPL", 240.0, "CALL", "2026-08-21", 1.0, 3.45, None, None),
        ("BTO AAPL 240C exp 2026-08-21 @3.45", "AAPL", 240.0, "CALL", "2026-08-21", 1.0, 3.45, None, None),
        ("AAPL 240C 8-21 @3.45", "AAPL", 240.0, "CALL", "2026-08-21", 1.0, 3.45, None, None),
        ("BTO AAPL 240C x2 @3.45", "AAPL", 240.0, "CALL", default_expiry, 2.0, 3.45, None, None),
        ("AAPL 240C 2x @3.45", "AAPL", 240.0, "CALL", default_expiry, 2.0, 3.45, None, None),
        ("AAPL 240C @3.45 SL below 2.20 PT 5.80", "AAPL", 240.0, "CALL", default_expiry, 1.0, 3.45, 2.20, 5.80),
        ("AAPL 240C stop 2.20 trim at 5.80", "AAPL", 240.0, "CALL", default_expiry, 1.0, None, 2.20, None),
        ("BTO AAPL 240C @3.45 take profit 5.80", "AAPL", 240.0, "CALL", default_expiry, 1.0, 3.45, None, 5.80),
        ("Buy lulu 290s 06/25 @9.35 BY 2.50 TP 9.30", "LULU", 290.0, "CALL", "2027-06-25", 1.0, 9.35, 2.50, 9.30),
    ]
    for text, root, strike, side, expiry, qty, fill, sl, target in cases:
        routed = classify_and_parse(text)
        _assert(routed.kind == "OPTION", f"{text} -> OPTION", f"got {routed.kind}")
        opt = routed.option
        _assert(opt is not None and opt.valid, f"{text} valid", getattr(opt, "reason", ""))
        _assert(opt is not None and opt.root == root, f"{text} root == {root}", f"got {getattr(opt, 'root', None)}")
        _assert(opt is not None and opt.strike == strike, f"{text} strike == {strike}", f"got {getattr(opt, 'strike', None)}")
        _assert(opt is not None and opt.side == side, f"{text} side == {side}", f"got {getattr(opt, 'side', None)}")
        _assert(opt is not None and opt.expiry_date == expiry, f"{text} expiry == {expiry}", f"got {getattr(opt, 'expiry_date', None)}")
        _assert(opt is not None and opt.quantity == qty, f"{text} qty == {qty}", f"got {getattr(opt, 'quantity', None)}")
        _assert(opt is not None and opt.fill_price == fill, f"{text} fill == {fill}", f"got {getattr(opt, 'fill_price', None)}")
        _assert(opt is not None and opt.stop_loss == sl, f"{text} SL == {sl}", f"got {getattr(opt, 'stop_loss', None)}")
        _assert(opt is not None and opt.target_price == target, f"{text} target == {target}", f"got {getattr(opt, 'target_price', None)}")

def test_btc_stc_shorthand() -> None:
    print("\nTest: BTO/STC shorthand triggers option detection")
    text = "BTO AAPL 150C"
    routed = classify_and_parse(text)
    _assert(routed.kind == "OPTION", "BTO AAPL 150C -> OPTION", f"got {routed.kind}")


def test_option_stop_target_parse() -> None:
    print("\nTest: option SL/TP fields parse correctly")
    opt = classify_and_parse("BTO AAPL 240C 08/21 @3.45 SL 2.20 TP 5.80 qty 1").option
    _assert(opt is not None and opt.valid, "BTO AAPL 240C valid")
    _assert(opt is not None and opt.root == "AAPL", "root == AAPL", f"got {getattr(opt, 'root', None)}")
    _assert(opt is not None and opt.strike == 240.0, "strike == 240.0", f"got {getattr(opt, 'strike', None)}")
    _assert(opt is not None and opt.fill_price == 3.45, "entry premium == 3.45", f"got {getattr(opt, 'fill_price', None)}")
    _assert(opt is not None and opt.stop_loss == 2.20, "SL == 2.20", f"got {getattr(opt, 'stop_loss', None)}")
    _assert(opt is not None and opt.target_price == 5.80, "TP == 5.80", f"got {getattr(opt, 'target_price', None)}")


def test_trailing_stop_percent_is_not_misread_as_an_absolute_stop_loss() -> None:
    print("\nTest: 'trailing stop N%' / 'trail stop N%' sets trailing_stop_pct only")
    for text in (
        "BTO AAPL 190C 9/19 @2.50 trailing stop 20%",
        "BTO NFLX 1450C 10/17 @18.40 TRAIL STOP 15%",
    ):
        opt = classify_and_parse(text).option
        _assert(opt is not None and opt.valid, f"{text!r} parses as a valid option")
        _assert(
            opt is not None and opt.stop_loss is None,
            f"{text!r}: stop_loss stays None, not the trailing percentage misread as a dollar price",
            f"got stop_loss={getattr(opt, 'stop_loss', None)}",
        )
        _assert(
            opt is not None and opt.trailing_stop_pct is not None,
            f"{text!r}: trailing_stop_pct is still captured correctly",
            f"got {getattr(opt, 'trailing_stop_pct', None)}",
        )


def test_numbered_pt_targets_alongside_tp_targets() -> None:
    print("\nTest: PT1/PT2 numbered targets are recognized like TP1/TP2")
    opt = classify_and_parse("BTO AAPL 190C 9/19 @2.50 PT1: 3.50 PT2: 4.50 SL: 1.80").option
    _assert(opt is not None and opt.valid, "parses as a valid option")
    _assert(
        opt is not None and opt.target_prices == (3.5, 4.5),
        "PT1/PT2 populate target_prices in order",
        f"got {getattr(opt, 'target_prices', None)}",
    )
    _assert(
        opt is not None and opt.stop_loss == 1.8,
        "SL still parses correctly alongside PT-style targets",
        f"got {getattr(opt, 'stop_loss', None)}",
    )


def test_multi_leg_no_strike() -> None:
    print("\nTest: multi-leg structure without a concrete strike/side is tracked-only")
    text = "Sell an iron condor on SPY"
    routed = classify_and_parse(text)
    _assert(routed.kind == "OPTION", "kind == OPTION", f"got {routed.kind}")
    opt = routed.option
    _assert(opt.is_multi_leg, "is_multi_leg True")
    _assert(opt.structure == "iron_condor", "structure == iron_condor", f"got {opt.structure}")
    _assert(opt.strike is None, "no strike parsed for pure multi-leg mention")


def test_executable_multi_leg_signals() -> None:
    print("\nTest: executable multi-leg Discord signals preserve every leg")
    cases = [
        (
            "BTO AAPL 240C / STO AAPL 250C 09/19 @4.60 Debit Qty 5",
            "AAPL", "bull_call_spread", 5.0, 4.60,
            [(240.0, "CALL", "open_long"), (250.0, "CALL", "open_short")],
        ),
        (
            "BTO TSM 250P / STO TSM 230P 10/17 @5.25 Debit Qty 3",
            "TSM", "bear_put_spread", 3.0, 5.25,
            [(250.0, "PUT", "open_long"), (230.0, "PUT", "open_short")],
        ),
        (
            "BUY SPY 640C + 640P 09/19 @8.20 Debit Qty 2",
            "SPY", "long_straddle", 2.0, 8.20,
            [(640.0, "CALL", "open_long"), (640.0, "PUT", "open_long")],
        ),
        (
            "BUY QQQ 600C + 570P 10/17 @7.10 Debit Qty 4",
            "QQQ", "long_strangle", 4.0, 7.10,
            [(600.0, "CALL", "open_long"), (570.0, "PUT", "open_long")],
        ),
    ]
    for text, root, structure, qty, debit, expected_legs in cases:
        option = classify_and_parse(text).option
        _assert(option is not None and option.valid, f"{root} multi-leg signal valid")
        _assert(option is not None and option.structure == structure, f"{root} structure == {structure}")
        _assert(option is not None and option.quantity == qty, f"{root} strategy quantity preserved")
        _assert(option is not None and option.fill_price == debit, f"{root} net debit preserved")
        _assert(option is not None and option.price_effect == "debit", f"{root} debit effect preserved")
        actual_legs = [(leg.strike, leg.side, leg.order_action) for leg in (option.legs if option else ())]
        _assert(actual_legs == expected_legs, f"{root} all leg actions preserved", str(actual_legs))


def test_no_trade_commentary() -> None:
    print("\nTest: pure market commentary routes to NO_TRADE, not INVALID")
    for text in (
        "ADX above 40 indicates a strong trend.",
        "No clear edge today.",
        "WAIT FOR CONFIRMATION",
        "Federal Reserve announced an unexpected interest rate hike.",
    ):
        routed = classify_and_parse(text)
        _assert(routed.kind == "NO_TRADE", f"'{text}' -> NO_TRADE", f"got {routed.kind}")


def test_invalid_only_for_empty() -> None:
    print("\nTest: INVALID is reserved for empty/unparseable input")
    _assert(classify_and_parse("").kind == "INVALID", "empty string -> INVALID")
    _assert(classify_and_parse("   ").kind == "INVALID", "whitespace-only -> INVALID")


def test_equity_path_unchanged() -> None:
    print("\nTest: plain equity signals still route through signal_parser unchanged")
    routed = classify_and_parse("buy AAPL qty 2")
    _assert(routed.kind == "EQUITY", "kind == EQUITY", f"got {routed.kind}")
    _assert(routed.equity is not None and routed.equity.action == "BUY", "action == BUY")
    _assert(routed.equity is not None and routed.equity.symbol == "AAPL", "symbol == AAPL")
    _assert(routed.equity is not None and routed.equity.quantity == 2.0, "quantity == 2.0")

    # Direct signal_parser.parse_signal must be completely untouched by this feature.
    direct = parse_signal("buy AAPL qty 2")
    _assert(direct.action == "BUY" and direct.symbol == "AAPL" and direct.quantity == 2.0,
            "signal_parser.parse_signal output unchanged")

    hold_text = "HOLD MSFT. Mixed technical indicators and earnings tomorrow."
    routed_hold = classify_and_parse(hold_text)
    _assert(routed_hold.kind == "EQUITY", "HOLD MSFT commentary -> EQUITY", f"got {routed_hold.kind}")
    _assert(routed_hold.equity is not None and routed_hold.equity.action == "HOLD", "HOLD MSFT action == HOLD")
    _assert(routed_hold.equity is not None and routed_hold.equity.symbol == "MSFT", "HOLD MSFT symbol == MSFT", f"got {getattr(routed_hold.equity, 'symbol', None)}")

    direct_hold = parse_signal(hold_text)
    _assert(direct_hold.action == "HOLD" and direct_hold.symbol == "MSFT",
            "signal_parser keeps MSFT when ticker has trailing period")


def test_conditional_equity_signals_become_watch_hold() -> None:
    print("\nTest: conditional equity signals become HOLD/watch, not immediate trades")
    cases = [
        ("BUY GOOGL if price closes above 205, otherwise HOLD.", "GOOGL"),
        ("SELL TSLA if price falls below 295.", "TSLA"),
    ]
    for text, symbol in cases:
        routed = classify_and_parse(text)
        _assert(routed.kind == "EQUITY", f"{text} -> EQUITY", f"got {routed.kind}")
        _assert(routed.equity is not None and routed.equity.symbol == symbol, f"{text} symbol == {symbol}", f"got {getattr(routed.equity, 'symbol', None)}")
        _assert(routed.equity is not None and routed.equity.action == "HOLD", f"{text} action == HOLD", f"got {getattr(routed.equity, 'action', None)}")
        _assert(routed.equity is not None and "Conditional setup" in routed.equity.reason, f"{text} has conditional reason")


def test_single_letter_ticker_is_not_an_option_side() -> None:
    print("\nTest: single-letter ticker C is distinct from the CALL suffix")
    cases = [
        ("BTO 7 C 120C 08/21 @10.75 LIMIT", "C", 120.0, "CALL", 7.0, "open_long"),
        ("STO 2 C 620P 08/21 @1.76 MARKET", "C", 620.0, "PUT", 2.0, "open_short"),
        ("STO 7 C 240P 08/21 @9.38 MARKET", "C", 240.0, "PUT", 7.0, "open_short"),
        ("STO 6 C 1000P 09/19 @18.56 MARKET", "C", 1000.0, "PUT", 6.0, "open_short"),
        ("BTO 3 XLE 200C 09/19 @14.97 MARKET", "XLE", 200.0, "CALL", 3.0, "open_long"),
        ("STC 8 XLF 450C 10/17 @5.88 LIMIT", "XLF", 450.0, "CALL", 8.0, "close_long"),
    ]
    for text, root, strike, side, qty, action in cases:
        routed = classify_and_parse(text)
        option = routed.option
        _assert(routed.kind == "OPTION" and option is not None and option.valid, f"{text} is valid")
        _assert(option is not None and option.root == root, f"{text} root == {root}")
        _assert(option is not None and option.strike == strike, f"{text} strike == {strike}")
        _assert(option is not None and option.side == side, f"{text} side == {side}")
        _assert(option is not None and option.quantity == qty, f"{text} qty == {qty}")
        _assert(option is not None and option.order_action == action, f"{text} action == {action}")
        _assert(option is not None and not option.is_multi_leg, f"{text} remains single-leg")


def test_conditional_option_lifecycle_fields() -> None:
    print("\nTest: conditional scale-in, timed exit, and risk override fields")
    avgo = classify_and_parse(
        "BTO AVGO 420C 11/21 @6.80 Qty 2. Add 3 more contracts above 425 breakout."
    ).option
    _assert(avgo is not None and avgo.valid, "AVGO conditional scale-in is valid")
    _assert(avgo is not None and avgo.quantity == 2, "AVGO base qty == 2")
    _assert(avgo is not None and avgo.add_quantity == 3, "AVGO add qty == 3")
    _assert(avgo is not None and avgo.add_trigger_underlying_direction == "above", "AVGO add direction == above")
    _assert(avgo is not None and avgo.add_trigger_underlying_price == 425, "AVGO add trigger == 425")

    nflx = classify_and_parse(
        "BTO NFLX 1450P 12/19 @18.60 Qty 2 EXIT ALL POSITIONS 30 MINUTES BEFORE MARKET CLOSE IF TARGET NOT HIT"
    ).option
    _assert(nflx is not None and nflx.valid, "NFLX timed exit is valid")
    _assert(nflx is not None and nflx.exit_before_market_close, "NFLX time exit enabled")
    _assert(nflx is not None and nflx.exit_minutes_before_close == 30, "NFLX exits 30 minutes before close")
    _assert(nflx is not None and nflx.exit_if_target_not_hit, "NFLX target-not-hit qualifier preserved")

    arm = classify_and_parse(
        "Starter on ARM 190C 10/17 @4.35. Looking to add over 195. Risking only 1% on this trade."
    ).option
    _assert(arm is not None and arm.valid, "ARM starter is valid")
    _assert(arm is not None and arm.position_type == "starter", "ARM position type == starter")
    _assert(arm is not None and arm.add_trigger_underlying_price == 195, "ARM add trigger == 195")
    _assert(arm is not None and arm.risk_stop_pct == 1, "ARM risk stop == 1%")


def test_combo_and_partial_close_are_multi_leg() -> None:
    """Regression test: is_multi_leg must count option+equity legs together, and
    must trust build_multi_leg_contract()'s own MULTI_LEG determination instead of
    re-deriving a leg count -- both a covered-call combo (1 option + 1 stock leg)
    and a partial-leg-close on an existing position (1 option leg only) are
    genuinely multi-leg strategies even though only one option leg is projected.
    """
    print("\nTest: combo (stock+option) and partial-leg-close signals are multi-leg")
    covered_call = classify_and_parse(
        "BUY 100 QQQ SHARES + STO 250C 09/19/2026 @4.27 CREDIT"
    ).option
    _assert(covered_call is not None and covered_call.valid, "covered call combo valid")
    _assert(
        covered_call is not None and covered_call.is_multi_leg,
        "covered call combo is_multi_leg == True",
        f"got {getattr(covered_call, 'is_multi_leg', None)}",
    )
    _assert(
        covered_call is not None and covered_call.contains_equity_leg,
        "covered call combo contains_equity_leg == True",
    )

    partial_close = classify_and_parse(
        "BTC ONLY THE SHORT TSLA 290C 10/16/2026; LEAVE ALL OTHER IRON CONDOR LEGS OPEN"
    ).option
    _assert(partial_close is not None and partial_close.valid, "partial leg close valid")
    _assert(
        partial_close is not None and partial_close.is_multi_leg,
        "partial leg close is_multi_leg == True",
        f"got {getattr(partial_close, 'is_multi_leg', None)}",
    )


def test_short_straddle_and_strangle_classification() -> None:
    """Regression test: options_parser's own leg-based classifier must label
    short (sold) straddles/strangles correctly, not fall through to generic
    'multi_leg' -- it previously only had the symmetric branch for long legs.
    """
    print("\nTest: short straddle/strangle structure classification")
    from .options_parser import ParsedOptionLeg, _classify_multi_leg_structure

    straddle_legs = (
        ParsedOptionLeg(root="AAPL", strike=150.0, side="CALL", order_action="open_short",
                        ratio_qty=1, expiry_date="2026-08-21"),
        ParsedOptionLeg(root="AAPL", strike=150.0, side="PUT", order_action="open_short",
                        ratio_qty=1, expiry_date="2026-08-21"),
    )
    _assert(
        _classify_multi_leg_structure(straddle_legs, None) == "short_straddle",
        "same-strike short legs classify as short_straddle",
    )

    strangle_legs = (
        ParsedOptionLeg(root="AAPL", strike=160.0, side="CALL", order_action="open_short",
                        ratio_qty=1, expiry_date="2026-08-21"),
        ParsedOptionLeg(root="AAPL", strike=140.0, side="PUT", order_action="open_short",
                        ratio_qty=1, expiry_date="2026-08-21"),
    )
    _assert(
        _classify_multi_leg_structure(strangle_legs, None) == "short_strangle",
        "different-strike short legs classify as short_strangle",
    )


def run_all() -> None:
    print("=" * 60)
    print("OPTIONS PARSER TEST HARNESS")
    print("No API calls -- pure parsing/formula tests")
    print("=" * 60)

    test_occ_symbol_known_answer()
    test_index_proxy_mapping()
    test_user_example_journal_single_leg()
    test_present_tense_new_order()
    test_natural_language_option_order()
    test_ce_pe_and_premium_formats()
    test_delta_option_signal()
    test_mixed_options_routing_matrix()
    test_real_world_discord_option_variants()
    test_btc_stc_shorthand()
    test_option_stop_target_parse()
    test_trailing_stop_percent_is_not_misread_as_an_absolute_stop_loss()
    test_numbered_pt_targets_alongside_tp_targets()
    test_multi_leg_no_strike()
    test_executable_multi_leg_signals()
    test_no_trade_commentary()
    test_invalid_only_for_empty()
    test_equity_path_unchanged()
    test_single_letter_ticker_is_not_an_option_side()
    test_conditional_option_lifecycle_fields()
    test_combo_and_partial_close_are_multi_leg()
    test_short_straddle_and_strangle_classification()

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    print("=" * 60)

    if FAIL == 0:
        print("\nALL OPTIONS PARSER TESTS PASSED")
    else:
        print(f"\nFAILED: {FAIL} test(s) did not pass. See details above.")
        raise SystemExit(1)


if __name__ == "__main__":
    run_all()



