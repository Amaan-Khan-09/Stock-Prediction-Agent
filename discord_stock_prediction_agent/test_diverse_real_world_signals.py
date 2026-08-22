"""Regression tests from a diverse, real-world-sourced signal stress test
across equity, single-leg options, and multi-leg options -- covers genuine
parsing bugs found by running many realistic trader phrasings (multi-leg
spreads, casual scale-in/trim language, rolls, calendars) through the actual
parser rather than only the pre-existing curated corpora.
"""
from __future__ import annotations

from .options_parser import classify_and_parse

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


def test_option_mention_without_a_strike_never_falls_back_to_equity() -> None:
    print("\nTest: mentioning calls/puts with no strike stays OPTION (invalid), never becomes an equity trade")
    # Regression: "EXITING" wasn't in the close-action token set, so
    # looks_like_option_signal's CALL/PUT fallback check was never reached,
    # and the message silently became a plain equity SELL -- exactly the
    # wrong asset class for a message that was never about the underlying.
    text = "Exiting AAPL calls into close, EOD flat"
    routed = classify_and_parse(text)
    _assert(routed.kind == "OPTION", f"{text!r} classified as OPTION, not EQUITY", f"got {routed.kind}")
    _assert(routed.equity is None, "no equity signal is produced for an options-only mention")
    _assert(
        routed.option is not None and not routed.option.valid,
        "correctly invalid (no strike given) rather than a fabricated trade",
    )


def test_word_fraction_partial_closes_set_close_percent() -> None:
    print("\nTest: 'half'/'a third'/'N of M' partial-exit phrasing sets close_percent")
    cases = [
        ("Trimming a third of my TSLA 250C position at 12.40, letting rest ride to 20", 33.34),
        ("Scaling out of half my NVDA calls at 8.20", 50.0),
        ("Sold half my AAPL 190C at 4.50", 50.0),
        ("Sold another half AAPL 190C at 5.00", 50.0),
        ("STC 2 of 5 MSFT 420C at 6.75", 40.0),
    ]
    for text, expected in cases:
        opt = classify_and_parse(text).option
        _assert(
            opt is not None and opt.close_percent == expected,
            f"{text!r} -> close_percent == {expected}",
            f"got {getattr(opt, 'close_percent', None)}",
        )
    # Existing numeric-percent phrasing must be unaffected by the new fallback.
    opt = classify_and_parse("STC 50% OF AAPL 240C 09/19 @6.80 HOLD REMAINING CONTRACTS").option
    _assert(opt is not None and opt.close_percent == 50, "existing 'STC 50%' phrasing still works unchanged")


def test_shared_root_propagates_to_legs_that_dont_repeat_the_ticker() -> None:
    print("\nTest: calendar/diagonal spreads and rolls correctly share one root across legs")
    # Regression: the multi-leg regex's optional per-leg root-token slot
    # captured the connector word ("TO") or a month abbreviation ("NOV",
    # "DEC") instead of falling back to the signal's real (shared) ticker,
    # since both are 1-6 all-caps letters just like a real symbol -- the
    # whole strategy was then wrongly rejected as spanning multiple
    # underlyings even though only one ticker was ever mentioned.
    cases = [
        ("Rolling TSLA 240C to 250C same exp for a .60 debit", "TSLA"),
        ("Bought NVDA Dec calendar: sell Nov 140C, buy Dec 140C for 2.10 debit", "NVDA"),
        ("Diagonal on AMD: STC Aug 150C, BTO Sep 155C for 1.10 net debit", "AMD"),
    ]
    for text, root in cases:
        opt = classify_and_parse(text).option
        _assert(opt is not None and opt.valid, f"{text!r} parses as valid", getattr(opt, "reason", None))
        if opt is not None and opt.valid:
            roots = {leg.root for leg in opt.legs}
            _assert(roots == {root}, f"{text!r}: both legs share root {root}", str(roots))


def test_bare_butterfly_keyword_and_compact_strikes_with_trailing_side_word() -> None:
    print("\nTest: bare 'butterfly'/'broken wing butterfly' keyword and a trailing side word (not glued to a strike)")
    # Regression: only "iron butterfly" and "butterfly spread" were
    # recognized structure keywords -- a bare "butterfly" (or the common
    # "broken wing butterfly" variant) matched nothing at all. Separately,
    # the compact 3-strike shorthand ("5800/5850/5900") only expanded into
    # 3 legs when a C/P letter was glued to one of the numbers (e.g.
    # "15P") -- a trailing, separate side word ("... calls") wasn't
    # recognized as a fallback.
    cases = [
        ("Broken wing butterfly on SPX 5800/5850/5900 calls for a small credit", "SPX", [5800.0, 5850.0, 5900.0], "CALL"),
        ("Bought AAPL butterfly 190/195/200 calls for .80 debit", "AAPL", [190.0, 195.0, 200.0], "CALL"),
    ]
    for text, root, strikes, side in cases:
        opt = classify_and_parse(text).option
        _assert(opt is not None and opt.valid, f"{text!r} parses as valid", getattr(opt, "reason", None))
        if opt is not None and opt.valid:
            _assert(len(opt.legs) == 3, f"{text!r}: expands to 3 legs", str(len(opt.legs)))
            _assert(
                sorted(leg.strike for leg in opt.legs) == strikes,
                f"{text!r}: correct strike triple", str(sorted(leg.strike for leg in opt.legs)),
            )
            _assert(all(leg.side == side for leg in opt.legs), f"{text!r}: all legs are {side}")
            _assert(all(leg.root == root for leg in opt.legs), f"{text!r}: all legs share root {root}")
    # The already-working glued-letter form (e.g. "7725/20/15P FLY") must
    # still work unchanged.
    opt = classify_and_parse("SPX : Bought SPX 7725/20/15P FLY at .80").option
    _assert(
        opt is not None and opt.valid and len(opt.legs) == 3,
        "pre-existing glued-side-letter FLY shorthand still works",
    )


def test_compact_two_strike_vertical_spread_expands_to_both_legs() -> None:
    print("\nTest: compact 2-strike vertical spread shorthand ('220/210 put spread') expands to both legs")
    # Regression: only the strike adjacent to the side word ("210 put")
    # matched the per-leg regex, so the whole strategy silently collapsed
    # into a single-leg close/open on that one strike, dropping the other
    # leg entirely -- turning a defined-risk 2-leg spread into a fabricated
    # naked single-leg trade.
    cases = [
        ("Selling AAPL 220/210 put spread for 3.20 credit", "AAPL", "PUT", "bull_put_spread", 220.0, "open_short", 210.0, "open_long"),
        ("Bought SPY 450/460 call spread for 2.50 debit", "SPY", "CALL", "bull_call_spread", 450.0, "open_long", 460.0, "open_short"),
        ("STO TSLA 250/240 put spread @1.80", "TSLA", "PUT", "bull_put_spread", 250.0, "open_short", 240.0, "open_long"),
    ]
    for text, root, side, structure, strike1, action1, strike2, action2 in cases:
        parsed = classify_and_parse(text)
        opt = parsed.option
        _assert(opt is not None and opt.valid, f"{text!r} parses as valid", getattr(opt, "reason", None))
        if opt is None or not opt.valid:
            continue
        _assert(len(opt.legs) == 2, f"{text!r}: expands to 2 legs", str(len(opt.legs)))
        _assert(opt.structure == structure, f"{text!r}: structure == {structure}", str(opt.structure))
        _assert(all(leg.root == root and leg.side == side for leg in opt.legs), f"{text!r}: both legs are {root} {side}")
        legs_by_strike = {leg.strike: leg.order_action for leg in opt.legs}
        _assert(legs_by_strike.get(strike1) == action1, f"{text!r}: {strike1} leg is {action1}", str(legs_by_strike))
        _assert(legs_by_strike.get(strike2) == action2, f"{text!r}: {strike2} leg is {action2}", str(legs_by_strike))
    # Prior FLY/butterfly/iron-condor compact shorthand must be unaffected.
    fly = classify_and_parse("SPX : Bought SPX 7725/20/15P FLY at .80").option
    _assert(fly is not None and fly.valid and len(fly.legs) == 3, "FLY compact shorthand still works unchanged")
    condor = classify_and_parse("SELL SPX 5800/5850/5900/5950 IC").option
    _assert(condor is not None and condor.valid and len(condor.legs) == 4, "iron condor compact shorthand still works unchanged")


def test_straddle_and_strangle_have_no_side_letter_by_definition() -> None:
    print("\nTest: straddle/strangle alerts (inherently no CALL/PUT side word) still parse as valid 2-leg trades")
    # Regression: a straddle/strangle is one call leg + one put leg by
    # definition, so it never has a per-leg (or even message-level) side
    # word at all -- the generic "no strike or delta with CALL/PUT side was
    # found" guard rejected every one of these outright, even though the
    # structure keyword itself was already being recognized correctly.
    text = "STO TSLA 250 straddle at 18.50"
    opt = classify_and_parse(text).option
    _assert(opt is not None and opt.valid, f"{text!r} parses as valid", getattr(opt, "reason", None))
    _assert(opt.structure == "short_straddle", f"{text!r}: structure == short_straddle", str(opt.structure))
    _assert(len(opt.legs) == 2, f"{text!r}: expands to 2 legs", str(len(opt.legs)))
    _assert(
        {leg.side for leg in opt.legs} == {"CALL", "PUT"} and all(leg.strike == 250.0 for leg in opt.legs),
        f"{text!r}: one CALL and one PUT leg, both at strike 250",
    )
    _assert(all(leg.order_action == "open_short" for leg in opt.legs), f"{text!r}: both legs sold to open (short straddle)")

    text = "Bought QQQ 400/410 strangle for 6.20"
    opt = classify_and_parse(text).option
    _assert(opt is not None and opt.valid, f"{text!r} parses as valid", getattr(opt, "reason", None))
    _assert(opt.structure == "long_strangle", f"{text!r}: structure == long_strangle", str(opt.structure))
    _assert(len(opt.legs) == 2, f"{text!r}: expands to 2 legs", str(len(opt.legs)))
    legs_by_side = {leg.side: leg.strike for leg in opt.legs}
    _assert(legs_by_side.get("PUT") == 400.0 and legs_by_side.get("CALL") == 410.0, f"{text!r}: put at the lower strike, call at the higher", str(legs_by_side))
    _assert(all(leg.order_action == "open_long" for leg in opt.legs), f"{text!r}: both legs bought to open (long strangle)")


def run_all() -> None:
    print("=" * 60)
    print("DIVERSE REAL-WORLD SIGNAL STRESS-TEST REGRESSIONS")
    print("=" * 60)

    test_option_mention_without_a_strike_never_falls_back_to_equity()
    test_word_fraction_partial_closes_set_close_percent()
    test_shared_root_propagates_to_legs_that_dont_repeat_the_ticker()
    test_bare_butterfly_keyword_and_compact_strikes_with_trailing_side_word()
    test_compact_two_strike_vertical_spread_expands_to_both_legs()
    test_straddle_and_strangle_have_no_side_letter_by_definition()

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    print("=" * 60)
    if FAIL == 0:
        print("\nALL DIVERSE REAL-WORLD SIGNAL TESTS PASSED")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    run_all()
