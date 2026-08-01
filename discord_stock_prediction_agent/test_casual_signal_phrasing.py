"""Regression tests for casual, natural-language trade phrasing.

Two related bug classes surfaced by stress-testing with fresh, unscripted
signals (not reused from any other test corpus):

1. Symbol/ticker collisions: several ordinary trading-syntax words (ALL, GO,
   MY, OUT, HALF, TARGET, BLOCK, SQUARE, ...) are themselves real tickers or
   company names in the live Alpaca symbol directory (or the hand-curated
   alias dictionaries). Left unguarded, "square off my TSLA position" or
   "bto arm ... if target not hit" silently resolved to the wrong company
   (SQ, TGT) instead of the actual named stock -- a valid=True trade against
   the wrong symbol, not a safe failure.
2. Gerund action words ("covering", "shorting") weren't recognized as
   BUY_TO_COVER/SELL_SHORT, so "covering my RIVN short now" read as opening a
   *new* short instead of closing the existing one -- the inverse of what was
   said.
"""
from __future__ import annotations

from .options_parser import _extract_root, classify_and_parse
from .signal_parser import parse_signal
from .stock_order_intent import parse_stock_order


def test_options_root_extraction_ignores_collision_words() -> None:
    # _extract_root is the shared root/underlying-symbol helper; check it
    # directly since not every phrase here also carries full option syntax
    # (strike/side) needed to classify as a complete OPTION signal.
    cases = [
        ("square off my tsla position", "TSLA"),
        ("bto arm 200c dec19 qty 2 exit before close if target not hit", "ARM"),
    ]
    for text, expected_root in cases:
        assert _extract_root(text) == expected_root, text

    option = classify_and_parse(
        "bto arm 200c dec19 qty 2 exit before close if target not hit"
    ).option
    assert option is not None and option.valid
    assert option.root == "ARM"


def test_equity_fallback_symbol_extraction_ignores_collision_words() -> None:
    cases = [
        ("go long on sofi, 30 shares", "SOFI", "BUY"),
        ("close my entire msft position", "MSFT", "SELL"),
        ("close out amzn", "AMZN", "SELL"),
        ("trim half my meta position", "META", "SELL"),
        ("sell all my aapl shares", "AAPL", "SELL"),
    ]
    for text, expected_symbol, expected_action in cases:
        parsed = parse_signal(text)
        assert parsed.valid, text
        assert parsed.symbol == expected_symbol, text
        assert parsed.action == expected_action, text


def test_gerund_cover_and_short_are_recognized() -> None:
    covering = parse_signal("covering my rivn short now")
    assert covering.valid
    assert covering.symbol == "RIVN"
    assert covering.action == "BUY_TO_COVER"

    shorting = parse_signal("shorting 15 shares of coin here")
    assert shorting.valid
    assert shorting.symbol == "COIN"
    assert shorting.action == "SELL_SHORT"

    # Unrelated words containing "cover"/"short" as a substring must not
    # false-positive into BUY_TO_COVER/SELL_SHORT.
    assert not parse_signal("the position was covered nicely").valid
    assert not parse_signal("i discovered a new stock today").valid


def test_go_long_and_go_short_prefixes_resolve_the_real_ticker() -> None:
    assert parse_stock_order("ENTER LONG RIVN 100 SHARES MARKET") == {
        "asset_type": "STOCK", "action": "BUY", "symbol": "RIVN",
        "quantity": 100, "order_type": "MARKET", "status": "VALID",
    }
    assert parse_stock_order("GO SHORT RIVN 100 SHARES MARKET") == {
        "asset_type": "STOCK", "action": "SELL_SHORT", "symbol": "RIVN",
        "quantity": 100, "order_type": "MARKET", "status": "VALID",
    }


def run_all() -> None:
    test_options_root_extraction_ignores_collision_words()
    test_equity_fallback_symbol_extraction_ignores_collision_words()
    test_gerund_cover_and_short_are_recognized()
    test_go_long_and_go_short_prefixes_resolve_the_real_ticker()
    print("CASUAL SIGNAL PHRASING TESTS PASSED")


if __name__ == "__main__":
    run_all()
