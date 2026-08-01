"""Deterministic 1,000+ case corpus for Discord and webhook trade signals."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date

from .options_parser import classify_and_parse
from .options_symbol import build_occ_symbol


@dataclass(frozen=True)
class Expected:
    text: str
    kind: str
    symbol: str = ""
    action: str = ""
    option_side: str = ""
    order_action: str = ""


EQUITY_SYMBOLS = (
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD",
    "ORCL", "CRM", "NFLX", "AVGO", "COST", "PEP", "DIS", "XOM",
    "JPM", "WMT", "DELL", "TKO",
)
OPTION_ROOTS = (
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD",
    "ORCL", "CRM", "NFLX", "AVGO", "COST", "SPY", "QQQ",
)


def _equity_cases() -> list[Expected]:
    cases: list[Expected] = []
    templates = {
        "BUY": (
            "BUY {s} QTY {q}",
            "Long ${s} {q} shares",
            "🟢 BUY **{s}** quantity: {q}",
            "Enter {s} with {q} shares",
            "Accumulate #{s} qty={q}",
        ),
        "SELL": (
            "SELL {s} QTY {q}",
            "🔴 SELL **{s}** quantity: {q}",
            "Exit {s} position, {q} shares",
            "Reduce #{s} qty={q}",
        ),
        "HOLD": (
            "HOLD {s}. Mixed indicators.",
            "Watch ${s}, no entry yet.",
            "Wait on #{s} until confirmation.",
        ),
    }
    for symbol_index, symbol in enumerate(EQUITY_SYMBOLS):
        qty = symbol_index % 9 + 1
        for action, variants in templates.items():
            for variant in variants:
                cases.append(Expected(variant.format(s=symbol, q=qty), "EQUITY", symbol, action))
        for action in ("buy", "sell", "hold"):
            cases.append(
                Expected(
                    json.dumps({"action": action, "ticker": symbol, "quantity": qty}),
                    "EQUITY",
                    symbol,
                    action.upper(),
                )
            )
        cases.append(
            Expected(
                f'```json\n{{"signal":"buy","symbol":"{symbol}","qty":{qty},"order_type":"market"}}\n```',
                "EQUITY",
                symbol,
                "BUY",
            )
        )
        cases.extend(
            (
                Expected(f"Short ${symbol} {qty} shares", "EQUITY", symbol, "SELL_SHORT"),
                Expected(f"\U0001f7e2 {symbol} qty {qty}", "EQUITY", symbol, "BUY"),
                Expected(f"\U0001f534 {symbol} qty {qty}", "EQUITY", symbol, "SELL"),
                Expected(f"\U0001f7e1 {symbol}", "EQUITY", symbol, "HOLD"),
            )
        )
    return cases


def _single_option_cases() -> list[Expected]:
    cases: list[Expected] = []
    actions = (
        ("BTO", "open_long"),
        ("STO", "open_short"),
        ("STC", "close_long"),
        ("BTC", "close_short"),
    )
    expiry = "2026-09-18"
    for root_index, root in enumerate(OPTION_ROOTS):
        for action_index, (action, order_action) in enumerate(actions):
            strike = 100 + root_index * 10 + action_index * 5
            qty = action_index + 1
            for side in ("C", "P"):
                full_side = "CALL" if side == "C" else "PUT"
                variants = (
                    f"{action} {root} {strike}{side} 09/18 @3.45 QTY {qty}",
                    f"{action} {qty} {root} {strike} {full_side} Sep 18 LIMIT 3.45",
                    f"🚨 {action} ${root} {strike}{side} 2026-09-18 @{3.45 + action_index:.2f}",
                    f"**{action}** {root} {strike} {'CE' if side == 'C' else 'PE'} 09-18 x{qty}",
                )
                for variant in variants:
                    cases.append(
                        Expected(variant, "OPTION", root, option_side=full_side, order_action=order_action)
                    )
                cases.append(
                    Expected(
                        json.dumps(
                            {
                                "action": action,
                                "underlying": root,
                                "strike_price": strike,
                                "right": full_side.lower(),
                                "expiration_date": expiry,
                                "contracts": qty,
                                "limit_price": 3.45,
                                "stop_loss": 2.2,
                                "take_profit": 5.8,
                            }
                        ),
                        "OPTION",
                        root,
                        option_side=full_side,
                        order_action=order_action,
                    )
                )
    return cases


def _occ_cases() -> list[Expected]:
    cases: list[Expected] = []
    expiry = date(2026, 9, 18)
    for root_index, root in enumerate(OPTION_ROOTS):
        for side in ("CALL", "PUT"):
            strike = 100 + root_index * 10
            occ = build_occ_symbol(root, expiry, side, strike)
            side_letter = "C" if side == "CALL" else "P"
            cases.extend(
                (
                    Expected(f"BTO O:{occ} QTY 1", "OPTION", root, option_side=side, order_action="open_long"),
                    Expected(f"STC .{occ} @5.20", "OPTION", root, option_side=side, order_action="close_long"),
                    Expected(f"BUY {occ}", "OPTION", root, option_side=side, order_action="open_long"),
                    Expected(
                        f"SELL TO OPEN {root} {strike}{side_letter} 2026-09-18",
                        "OPTION",
                        root,
                        option_side=side,
                        order_action="open_short",
                    ),
                )
            )
    return cases


def _multi_leg_cases() -> list[Expected]:
    cases: list[Expected] = []
    roots = OPTION_ROOTS[:10]
    for index, root in enumerate(roots):
        low = 100 + index * 10
        high = low + 10
        expiry = "09/18"
        cases.extend(
            (
                Expected(
                    f"BTO {root} {low}C / STO {root} {high}C {expiry} @4.60 Debit Qty 2",
                    "OPTION",
                    root,
                ),
                Expected(
                    f"BTO {root} {high}P / STO {root} {low}P {expiry} @5.10 Debit Qty 3",
                    "OPTION",
                    root,
                ),
                Expected(
                    f"BUY {root} {high}C + {high}P {expiry} @8.20 Debit Qty 2",
                    "OPTION",
                    root,
                ),
                Expected(
                    f"STO {root} {low}P / BTO {root} {low - 5}P / "
                    f"STO {root} {high}C / BTO {root} {high + 5}C {expiry} @2.10 Credit Qty 2",
                    "OPTION",
                    root,
                ),
                Expected(
                    json.dumps(
                        {
                            "underlying": root,
                            "expiration": "2026-09-18",
                            "quantity": 2,
                            "price": 4.6,
                            "price_effect": "debit",
                            "legs": [
                                {"action": "buy_to_open", "strike": low, "right": "call"},
                                {"action": "sell_to_open", "strike": high, "right": "call"},
                            ],
                        }
                    ),
                    "OPTION",
                    root,
                ),
            )
        )
    return cases


def _commentary_cases() -> list[Expected]:
    messages = (
        "No clear edge today.",
        "Waiting for CPI before taking risk.",
        "Market breadth is mixed and volatility remains elevated.",
        "Watching the open before making a decision.",
        "Earnings season could increase volatility.",
        "The index is consolidating near resistance.",
        "No trade until confirmation.",
        "Risk management matters more than prediction.",
        "Volume is lower than yesterday.",
        "Federal Reserve meeting is tomorrow.",
    )
    return [Expected(f"{message} #{index}", "NO_TRADE") for index in range(100) for message in messages]


def build_corpus() -> list[Expected]:
    corpus = [
        *_equity_cases(),
        *_single_option_cases(),
        *_occ_cases(),
        *_multi_leg_cases(),
        *_commentary_cases(),
    ]
    corpus.extend(
        (
            Expected(
                '{"action":"buy_to_open","symbol":"AAPL","strike":240,'
                '"right":"call","expiration":"2026-09-18","contracts":2,'
                '"type":"limit","price":3.45}',
                "OPTION",
                "AAPL",
                option_side="CALL",
                order_action="open_long",
            ),
            Expected(
                '{"message":"SELL MSFT QTY 3"}',
                "EQUITY",
                "MSFT",
                "SELL",
            ),
        )
    )
    return corpus


def run_all() -> None:
    corpus = build_corpus()
    assert len(corpus) >= 1_000, f"corpus too small: {len(corpus)}"
    failures: list[str] = []
    for expected in corpus:
        parsed = classify_and_parse(expected.text)
        if parsed.kind != expected.kind:
            failures.append(f"kind {parsed.kind} != {expected.kind}: {expected.text}")
            continue
        if expected.kind == "EQUITY":
            equity = parsed.equity
            if not equity or not equity.valid:
                failures.append(f"invalid equity: {expected.text}")
            elif equity.symbol != expected.symbol or equity.action != expected.action:
                failures.append(
                    f"equity {equity.symbol}/{equity.action} != "
                    f"{expected.symbol}/{expected.action}: {expected.text}"
                )
        elif expected.kind == "OPTION":
            option = parsed.option
            if not option or not option.valid:
                failures.append(f"invalid option: {expected.text} ({getattr(option, 'reason', '')})")
            elif option.root != expected.symbol:
                failures.append(f"option root {option.root} != {expected.symbol}: {expected.text}")
            elif expected.option_side and option.side != expected.option_side:
                failures.append(f"option side {option.side} != {expected.option_side}: {expected.text}")
            elif expected.order_action and option.order_action != expected.order_action:
                failures.append(
                    f"option action {option.order_action} != {expected.order_action}: {expected.text}"
                )
    if failures:
        preview = "\n".join(failures[:30])
        raise AssertionError(f"{len(failures)}/{len(corpus)} corpus failures:\n{preview}")
    print(f"REAL-WORLD SIGNAL CORPUS PASSED: {len(corpus)} deterministic cases")


if __name__ == "__main__":
    run_all()
