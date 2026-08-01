"""
Tastytrade Delta-Mode Backtest Smoke Test.

Runs a minimal delta-mode backtest outside Streamlit to isolate payload issues.
Auth uses the same service layer as the Streamlit app (REFRESH_TOKEN flow).

Usage:
    python verify_tastytrade_delta_backtest.py [--symbol SPY] [--delta 50] [--dte 30]
    python verify_tastytrade_delta_backtest.py --symbol DELL --delta 30 --dte 45

Security:
- Never prints full token values.
- Masks all tokens to first 4 + ... + last 4 characters.
"""
from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path
from datetime import date, timedelta

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)


def _mask(token: str) -> str:
    if not token:
        return "(empty)"
    if len(token) <= 10:
        return "***"
    return f"{token[:4]}...{token[-4:]}"


def _section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tastytrade backtester smoke test — verifies all modes live")
    parser.add_argument("--symbol",         default="SPY",        help="Stock symbol (default: SPY)")
    parser.add_argument("--direction",      default="short",      help="long/short (default: short)")
    parser.add_argument("--side",           default="put",        help="call or put (default: put)")
    parser.add_argument("--delta",          type=int, default=50, help="Delta 1-99 (default: 50)")
    parser.add_argument("--dte",            type=int, default=30, help="Days to expiration (default: 30)")
    parser.add_argument("--quantity",       type=int, default=1,  help="Contracts (default: 1)")
    parser.add_argument(
        "--strike-mode",
        default="delta",
        choices=["delta", "percentageOtm", "priceOffset", "premium"],
        help="Strike selection mode to test (default: delta — confirmed working)",
    )
    parser.add_argument("--otm-pct",        type=float, default=5.0,  help="OTM%% for percentageOtm mode (default: 5.0)")
    parser.add_argument("--price-offset",   type=float, default=5.0,  help="$ offset for priceOffset mode (default: 5.0)")
    parser.add_argument("--premium",        type=float, default=1.0,  help="Target premium for premium mode (default: 1.0)")
    parser.add_argument(
        "--entry-frequency",
        default="every day",
        choices=["every day", "on_exact_dte_match", "weekly", "monthly"],
        help="Entry frequency to test (default: 'every day' — confirmed working; 'on_exact_dte_match' also confirmed via AMC run 3761BEE3)",
    )
    args = parser.parse_args()

    symbol = args.symbol.upper()
    _dir_map = {"buy": "long", "sell": "short", "long": "long", "short": "short"}
    direction_api = _dir_map.get(args.direction.lower(), args.direction.lower())
    strike_mode    = args.strike_mode
    entry_freq     = args.entry_frequency

    _section("1. ENVIRONMENT CONFIG")
    from src.config.settings import settings
    api_url = settings.tastytrade_api_base_url
    bt_url  = settings.tastytrade_backtester_base_url
    print(f"TASTYTRADE_API_BASE_URL      = {api_url}")
    print(f"TASTYTRADE_BACKTESTER_BASE_URL = {bt_url}")

    _section("2. AUTH TRUTH")
    from verify_tastytrade_auth_truth import verify_tastytrade_auth_truth
    truth = verify_tastytrade_auth_truth()
    print(f"Credential Source     : {truth.get('credential_source')}")
    print(f"Access Token Present  : {truth.get('access_token_present')}")
    print(f"Access Token (masked) : {truth.get('access_token_masked')}")
    print(f"Refresh Token Present : {truth.get('refresh_token_present')}")
    print(f"Refresh Token (masked): {truth.get('refresh_token_masked')}")
    print(f"Token Refresh Status  : {truth.get('token_refresh_status')}")
    print(f"Customer Check Status : {truth.get('customer_check_status')}")
    print(f"Auth HTTP Status      : {truth.get('auth_http_status')}")
    print(f"Backtest Allowed      : {truth.get('backtest_allowed')}")
    print(f"Reason                : {truth.get('reason')}")
    print(f"Secrets Masked        : {truth.get('secrets_masked')}")

    if not truth.get("backtest_allowed"):
        print("\n[BLOCKED] Backtest not allowed — check credentials.")
        sys.exit(1)

    _section("3. BACKTEST DATE WINDOW")
    today = date.today()
    end_d   = today - timedelta(days=5)
    start_d = end_d - timedelta(days=args.dte + 10)
    start_str = start_d.strftime("%Y-%m-%d")
    end_str   = end_d.strftime("%Y-%m-%d")
    print(f"Symbol         : {symbol}")
    print(f"Start          : {start_str}")
    print(f"End            : {end_str}")
    print(f"Direction      : {args.direction} -> API: {direction_api}")
    print(f"Side           : {args.side}")
    print(f"DTE            : {args.dte}")
    print(f"Quantity       : {args.quantity}")
    print(f"Strike Mode    : {strike_mode}  (delta=confirmed | percentageOtm/priceOffset/premium=pending verification)")
    print(f"Entry Freq     : {entry_freq}  (every day=confirmed | on_exact_dte_match=confirmed | weekly/monthly=likely)")

    _section("4. BUILD PAYLOAD")
    from src.services.tastytrade_backtester_service import (
        build_custom_legs_payload, create_backtest, poll_backtest,
        parse_backtest_result, validate_backtest_success,
    )

    # Build leg dict for the chosen strike mode
    leg: dict = {
        "type":                "equity-option",
        "direction":           direction_api,
        "quantity":            args.quantity,
        "side":                args.side.lower(),
        "daysUntilExpiration": args.dte,
        "strikeSelection":     strike_mode,
    }
    if strike_mode == "delta":
        leg["delta"] = args.delta
    elif strike_mode == "percentageOtm":
        leg["percentageOtm"] = args.otm_pct
        print(f"  percentageOtm  : {args.otm_pct}%")
    elif strike_mode == "priceOffset":
        leg["priceOffset"] = args.price_offset
        print(f"  priceOffset    : ${args.price_offset}")
    elif strike_mode == "premium":
        leg["premium"] = args.premium
        print(f"  premium        : ${args.premium}")

    payload = build_custom_legs_payload(symbol, start_str, end_str, [leg], entry_frequency=entry_freq)
    payload_dict = payload.to_dict()
    print(json.dumps(payload_dict, indent=2, default=str))

    _section("5. CREATE BACKTEST")
    backtest_id, create_err = create_backtest(payload)
    if create_err:
        print(f"[ERROR] create_backtest returned error: {create_err!r}")
        if create_err.startswith("BACKTEST_HTTP_"):
            parts = create_err.split(":", 1)
            code  = parts[0].replace("BACKTEST_HTTP_", "")
            body  = parts[1] if len(parts) > 1 else "(no body)"
            print(f"\nHTTP Status   : {code}")
            print(f"Response Body :\n{body}")
            print(f"\nStrike mode tested  : {strike_mode}")
            print(f"Entry freq tested   : {entry_freq}")
            print("\n[DIAGNOSIS] Auth succeeded but Tastytrade rejected the payload.")
            print("  If strike mode != 'delta'   → the strikeSelection field name may be wrong.")
            print("  If entry freq not in ('every day', 'on_exact_dte_match') → the frequency string may be wrong.")
            print("  If delta mode failed         → check delta range (1-99), DTE (1-365), symbol.")
        elif create_err.startswith("RATE_LIMITED:429:"):
            ra = create_err.split("retry_after=")[-1]
            print(f"[RATE_LIMITED] HTTP 429 — Retry-After: {ra}. Wait before retrying.")
        sys.exit(1)

    print(f"Backtest ID: {backtest_id}  [CREATE SUCCESS]")

    _section("6. POLL BACKTEST")
    bt_data, poll_err = poll_backtest(backtest_id)
    if poll_err or not bt_data:
        print(f"[ERROR] poll_backtest failed: {poll_err!r}")
        sys.exit(1)

    print(f"Poll status: {bt_data.get('status')}")

    _section("7. PARSE + VALIDATE")
    result     = parse_backtest_result(backtest_id, bt_data)
    validation = validate_backtest_success(result)

    print(f"Leg type         : {result.leg_type}")
    print(f"Trials           : {len(result.trials)}")
    print(f"Validation passed: {validation.passed}")
    if not validation.passed:
        for r in validation.reasons:
            print(f"  Reason: {r}")
    else:
        stats = result.statistics
        print(f"Total P&L   : ${float(stats.total_profit_loss):+,.2f}")
        print(f"Win Rate    : {stats.win_rate*100:.1f}%")
        print(f"Avg P&L     : ${float(stats.average_profit_loss):+,.2f}")
        print(f"Num Trades  : {stats.num_trades}")

    _section("8. SUMMARY")
    if validation.passed:
        print(f"RESULT: SUCCESS — strike_mode='{strike_mode}' + entry_frequency='{entry_freq}' VERIFIED WORKING.")
        print(f"  Trades: {result.statistics.num_trades}  Total P&L: ${float(result.statistics.total_profit_loss):+,.2f}")
        print("")
        print("VERIFICATION STATUS after this run:")
        print(f"  strike_mode='{strike_mode}' : VERIFIED ✓")
        print(f"  entry_frequency='{entry_freq}' : VERIFIED ✓")
        if strike_mode == "delta":
            print("  → To verify other modes, run:")
            print("      python verify_tastytrade_delta_backtest.py --strike-mode percentageOtm --otm-pct 5.0")
            print("      python verify_tastytrade_delta_backtest.py --strike-mode priceOffset --price-offset 5.0")
            print("      python verify_tastytrade_delta_backtest.py --strike-mode premium --premium 1.0")
            print("      python verify_tastytrade_delta_backtest.py --entry-frequency weekly")
            print("      python verify_tastytrade_delta_backtest.py --entry-frequency monthly")
    else:
        print(f"RESULT: VALIDATION FAILED — strike_mode='{strike_mode}' + entry_frequency='{entry_freq}' REJECTED.")
        print("  If leg_type='unknown'           → wrong equity-option type value (should not happen).")
        print("  If statistics=null + trials=[]  → Tastytrade API rejected field names.")
        print("  If all P&L=0                    → no contracts matched the strike/DTE combo.")
        print("")
        print("  Most likely field name issues to check:")
        print("    percentageOtm → try 'percent_otm', 'otm_percentage', 'pct_otm'")
        print("    priceOffset   → try 'price_offset', 'strike_offset', 'offset'")
        print("    premium       → try 'premium_target', 'target_premium', 'targetPremium'")
        print("    entry freq    → try 'daily', 'day', 'once_per_week', 'once_per_month'")


if __name__ == "__main__":
    main()
