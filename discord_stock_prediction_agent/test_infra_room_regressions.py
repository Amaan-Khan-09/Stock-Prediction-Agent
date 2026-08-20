"""Known-answer regression tests from a real trading-alert-room corpus.

A real Discord trading-alert room feed (SPX 0DTE scalps, percentage-based
partial exits, "roll"/"runners" jargon) was run through classify_and_parse()
and surfaced several real bugs, fixed alongside this file:

  1. "Sold"/"Closed"/"Exited" (past-tense) option signals fell through to
     order_action="unknown", which _option_order_route() in discord_agent.py
     silently defaults to a BUY -- so a message reporting a position being
     SOLD would have submitted an opposite-direction BUY order.
  2. A bare CALL/PUT side-letter left over after strike digits were
     stripped out by the root-token scan (e.g. the "C" in "7770C") could
     leak through as a guessed root symbol, and "C" happens to be a real
     ticker (Citigroup) -- so a message with no real root mention could
     silently misroute to the wrong company.
  3. The root-token scan capped words at 6 characters, splitting longer
     words ("ALREADY") into fragments ("ALREAD" + "Y"); the leftover
     fragment could itself look like a plausible ticker.
  4. The bare word "roll" forced *any* message into strict multi-leg
     parsing regardless of how many option legs were actually present,
     so single-leg entries merely narrating profit-recycling ("Bought
     SPX 7690C ... Roll using 20% Profit") were rejected outright.
  5. Cash-settled index roots (SPX, NDX, RUT, DJX, VIX, XSP) and options
     jargon (ITM/OTM/ATM) aren't excluded from signal_parser.py's equity
     ticker guesser, so a bare "SPX" mention with no strike/side could
     resolve through the symbol cache to an unrelated real equity ticker
     (SPX -> SPXC, "SPX Technologies") instead of correctly falling
     through to NO_TRADE.

Run (from the project root):
    venv\\Scripts\\python.exe -m discord_stock_prediction_agent.test_infra_room_regressions
"""
from __future__ import annotations

from .daily_signal_context import (
    classify_and_parse_with_daily_context,
    clear_all as clear_daily_context,
)
from .options_parser import classify_and_parse
from .signal_parser import _extract_symbol

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


def test_past_tense_close_maps_to_close_long() -> None:
    print("\nTest: 'Sold'/'Closed'/'Exited' resolve to close_long, not unknown")
    cases = [
        "Sold 50% SPX 7565C at 5.80",
        "SPX 7535P runners closed ITM at 18.15",
        "Sold SPX 7700P runner at 6.20 Holding 5",
        "Exited SPX 7680P at 4.50",
    ]
    for text in cases:
        option = classify_and_parse(text).option
        _assert(option is not None and option.valid, f"{text!r} parses as valid OPTION")
        _assert(
            option is not None and option.order_action == "close_long",
            f"{text!r} order_action == close_long",
            f"got {getattr(option, 'order_action', None)}",
        )


def test_bought_journal_tense_unchanged() -> None:
    print("\nTest: 'Bought' journal-entry tense is untouched (regression guard)")
    text = "Bought SPX 7570C at 1.40 EOD Lotto. Strangle is complete."
    option = classify_and_parse(text).option
    _assert(option is not None and option.valid, "still parses as valid OPTION")
    _assert(option is not None and option.tense == "past", "tense stays 'past'", f"got {getattr(option, 'tense', None)}")


def test_bare_side_letter_does_not_leak_as_root() -> None:
    print("\nTest: leftover CALL/PUT side-letter isn't guessed as the root")
    option = classify_and_parse("Sold 70% 7770C at 4 Already 100%").option
    _assert(option is not None, "still routes to OPTION kind")
    _assert(option.root == "", "root left empty rather than guessing 'C'", f"got {option.root!r}")
    _assert(not option.valid, "correctly fails closed instead of misrouting")


def test_explicit_single_letter_root_still_works() -> None:
    print("\nTest: an explicitly-stated single-letter root (real regression guard)")
    option = classify_and_parse("BTO 7 C 120C 08/21 @10.75 LIMIT").option
    _assert(option is not None and option.valid, "still valid")
    _assert(option is not None and option.root == "C", "root == C", f"got {getattr(option, 'root', None)}")


def test_long_word_truncation_does_not_leak_fragment() -> None:
    print("\nTest: words longer than 6 chars aren't sliced into a fake ticker")
    option = classify_and_parse("Sold most SPX 7770C at 5.20. Too many roll already").option
    _assert(option is not None and option.root == "SPX", "root correctly SPX, not a truncated fragment", f"got {getattr(option, 'root', None)}")


def test_roll_commentary_single_leg_not_forced_multi_leg() -> None:
    print("\nTest: single-leg 'roll' commentary no longer rejected as incomplete multi-leg")
    option = classify_and_parse("SPX : Bought SPX 7690C at 2.50 Roll using 20% Profit").option
    _assert(option is not None and option.valid, "parses as valid single-leg OPTION")
    _assert(option is not None and not option.is_multi_leg, "stays single-leg")
    _assert(option is not None and option.root == "SPX", "root == SPX")
    _assert(option is not None and option.strike == 7690.0, "strike == 7690.0")


def test_roll_two_leg_still_builds_multi_leg() -> None:
    print("\nTest: a genuine 2-leg ROLL X TO Y instruction still builds multi-leg (no regression)")
    option = classify_and_parse(
        "ROLL NVDA 190C 09/19 TO NVDA 205C 10/17 FOR 1.40 Debit"
    ).option
    _assert(option is not None and option.valid, "valid")
    _assert(option is not None and option.is_multi_leg, "still recognized as multi-leg")


def test_index_root_mention_does_not_resolve_to_equity() -> None:
    print("\nTest: bare index-root mentions don't resolve to an unrelated equity ticker")
    cases = [
        "SPX Holding 2 runners if get ITM",
        "Small Trade Need to hold 7735",
        "Better hold 7725",
    ]
    for text in cases:
        symbol = _extract_symbol(text)
        _assert(symbol == "", f"{text!r} -> no equity symbol guessed", f"got {symbol!r}")
        kind = classify_and_parse(text).kind
        _assert(kind == "NO_TRADE", f"{text!r} classifies as NO_TRADE", f"got {kind}")


def test_daily_context_fills_missing_root_same_day() -> None:
    print("\nTest: same-day channel memory fills a missing root")
    clear_daily_context()
    channel = "test-channel-1"
    first = classify_and_parse_with_daily_context("SPX : Bought SPX 7665C at 4.80", channel)
    _assert(first.kind == "OPTION" and first.option.valid, "first message parses normally")

    followup = classify_and_parse_with_daily_context(
        "Sold 70% 7770C at 4 Already 100%", channel
    )
    _assert(followup.kind == "OPTION" and followup.option is not None, "follow-up routes to OPTION")
    _assert(
        followup.option is not None and followup.option.valid and followup.option.root == "SPX",
        "follow-up root filled in from same-day memory",
        f"got root={getattr(followup.option, 'root', None)} valid={getattr(followup.option, 'valid', None)}",
    )


def test_daily_context_never_overrides_explicit_root() -> None:
    print("\nTest: same-day memory never overrides a root the message itself states")
    clear_daily_context()
    channel = "test-channel-2"
    classify_and_parse_with_daily_context("SPX : Bought SPX 7665C at 4.80", channel)
    other = classify_and_parse_with_daily_context("BTO AAPL 240C 09/18 @3.45 QTY 2", channel)
    _assert(
        other.kind == "OPTION" and other.option is not None and other.option.root == "AAPL",
        "explicitly-stated root (AAPL) is never overridden by remembered SPX",
        f"got {getattr(other.option, 'root', None)}",
    )


def test_daily_context_isolated_per_channel() -> None:
    print("\nTest: same-day memory is scoped per channel, not global")
    clear_daily_context()
    classify_and_parse_with_daily_context("SPX : Bought SPX 7665C at 4.80", "channel-A")
    followup = classify_and_parse_with_daily_context(
        "Sold 70% 7770C at 4 Already 100%", "channel-B"
    )
    _assert(
        followup.option is not None and not followup.option.valid,
        "a different channel does not inherit channel-A's remembered root",
    )


def test_daily_context_clears_on_new_day() -> None:
    print("\nTest: memory is discarded once the UTC date rolls over")
    from . import daily_signal_context as dsc

    clear_daily_context()
    channel = "test-channel-rollover"
    classify_and_parse_with_daily_context("SPX : Bought SPX 7665C at 4.80", channel)
    _assert(dsc.get_default_root(channel) == "SPX", "root remembered for today")

    dsc._context[channel]["date"] = "2000-01-01"
    _assert(dsc.get_default_root(channel) == "", "stale-day memory is discarded on read")

    followup = classify_and_parse_with_daily_context(
        "Sold 70% 7770C at 4 Already 100%", channel
    )
    _assert(
        followup.option is not None and not followup.option.valid,
        "no stale-day root is applied to a new day's ambiguous message",
    )


def run_all() -> None:
    print("=" * 60)
    print("TRADING-ALERT-ROOM REGRESSION TEST HARNESS")
    print("Known-answer cases from a real Discord alert-room corpus")
    print("=" * 60)

    test_past_tense_close_maps_to_close_long()
    test_bought_journal_tense_unchanged()
    test_bare_side_letter_does_not_leak_as_root()
    test_explicit_single_letter_root_still_works()
    test_long_word_truncation_does_not_leak_fragment()
    test_roll_commentary_single_leg_not_forced_multi_leg()
    test_roll_two_leg_still_builds_multi_leg()
    test_index_root_mention_does_not_resolve_to_equity()
    test_daily_context_fills_missing_root_same_day()
    test_daily_context_never_overrides_explicit_root()
    test_daily_context_isolated_per_channel()
    test_daily_context_clears_on_new_day()

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    print("=" * 60)

    if FAIL == 0:
        print("\nALL TRADING-ALERT-ROOM REGRESSION TESTS PASSED")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    run_all()
