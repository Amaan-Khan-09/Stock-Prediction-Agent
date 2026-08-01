"""
verify_tastytrade_auth_truth.py -- Tastytrade Auth Truth Verification

Standalone CLI + importable module.

Credential resolution order:
  1. FORCE_DISABLE_TASTYTRADE=true  -> MISSING_FORCED, backtest_allowed=false
  2. TASTYTRADE_REFRESH_TOKEN       -> REFRESH_TOKEN path (preferred)
  3. TASTYTRADE_ACCESS_TOKEN only   -> ACCESS_TOKEN_ONLY path (may be stale)
  4. Neither present                -> MISSING, backtest_allowed=false

Security rules:
  - Never expose token values in output.
  - Show only masked preview (abc...xyz).
  - No mock / fake / fallback output.
  - All backtest_allowed=true requires a real /customers/me HTTP 200.
"""
from __future__ import annotations

import datetime
import json
import os
import sys
from typing import Optional, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()

_TT_BASE = os.getenv("TASTYTRADE_API_BASE_URL", "https://api.tastyworks.com")
_TT_UA   = os.getenv("TASTYTRADE_USER_AGENT", "flex-ai-finance/1.0")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mask(token: str) -> str:
    """Return masked preview — never expose full token value."""
    if not token:
        return None
    if len(token) < 8:
        return "***"
    return f"{token[:3]}...{token[-4:]}"


def _bearer(token: str) -> str:
    t = token.strip()
    return t if t.startswith("Bearer ") else f"Bearer {t}"


def _exchange_refresh(refresh_token: str) -> Tuple[Optional[str], int, str]:
    """POST /oauth/token with grant_type=refresh_token.

    Returns (access_token_or_None, http_status, error_string).
    Token value is never logged or exposed in error messages.
    """
    client_secret = os.getenv("TASTYTRADE_CLIENT_SECRET", "").strip()
    payload: dict = {"grant_type": "refresh_token", "refresh_token": refresh_token}
    if client_secret:
        payload["client_secret"] = client_secret

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent":   _TT_UA,
    }
    try:
        r = requests.post(
            f"{_TT_BASE}/oauth/token",
            data=payload,
            headers=headers,
            timeout=15,
        )
        if r.status_code == 200:
            data  = r.json()
            token = data.get("access_token") or data.get("data", {}).get("access-token", "")
            if token:
                return token, 200, ""
            return None, 200, "HTTP 200 but access_token missing from response"
        return None, r.status_code, f"Token refresh failed: HTTP {r.status_code}"
    except Exception as exc:
        return None, 0, f"Token refresh error: {type(exc).__name__}"


def _check_customers_me(access_token: str) -> Tuple[bool, int, str]:
    """GET /customers/me.

    Returns (ok, http_status, error_string).
    """
    headers = {
        "Authorization": _bearer(access_token),
        "Content-Type":  "application/json",
        "User-Agent":    _TT_UA,
    }
    url = f"{_TT_BASE}/customers/me"
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            return True, 200, ""
        return False, r.status_code, f"Customer check failed: HTTP {r.status_code}"
    except Exception as exc:
        return False, 0, f"Customer check error: {type(exc).__name__}"


# ── Main function ─────────────────────────────────────────────────────────────

def verify_tastytrade_auth_truth() -> dict:
    """Live Tastytrade auth truth check.

    Returns structured dict.  Token values are never included — only masked
    previews and boolean presence flags.

    backtest_allowed=True only if:
      - FORCE_DISABLE_TASTYTRADE is not 'true', AND
      - either refresh or access token is present, AND
      - /customers/me returns HTTP 200 with that token.
    """
    now_utc = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    # ── FORCE_DISABLE check (highest priority) ────────────────────────────────
    force_disable = os.getenv("FORCE_DISABLE_TASTYTRADE", "false").lower() == "true"
    if force_disable:
        return {
            "credential_source":         "MISSING_FORCED",
            "access_token_present":      False,
            "refresh_token_present":     False,
            "access_token_masked":       None,
            "refresh_token_masked":      None,
            "token_refresh_attempted":   False,
            "token_refresh_status":      "NOT_RUN",
            "token_refresh_http_status": None,
            "customer_check_attempted":  False,
            "customer_check_status":     "NOT_RUN",
            "auth_http_status":          None,
            "backtest_allowed":          False,
            "reason": "FORCE_DISABLE_TASTYTRADE=true — tastytrade disabled for verification mode",
            "timestamp_utc":             now_utc,
            "secrets_masked":            True,
        }

    rt = os.getenv("TASTYTRADE_REFRESH_TOKEN", "").strip()
    at = os.getenv("TASTYTRADE_ACCESS_TOKEN", "").strip()

    result: dict = {
        "credential_source":         None,
        "access_token_present":      bool(at),
        "refresh_token_present":     bool(rt),
        "access_token_masked":       _mask(at),
        "refresh_token_masked":      _mask(rt),
        "token_refresh_attempted":   False,
        "token_refresh_status":      "NOT_RUN",
        "token_refresh_http_status": None,
        "customer_check_attempted":  False,
        "customer_check_status":     "NOT_RUN",
        "auth_http_status":          None,
        "backtest_allowed":          False,
        "reason":                    "",
        "timestamp_utc":             now_utc,
        "secrets_masked":            True,
    }

    # ── Both missing ──────────────────────────────────────────────────────────
    if not rt and not at:
        result["credential_source"] = "MISSING"
        result["reason"] = (
            "No TASTYTRADE_REFRESH_TOKEN or TASTYTRADE_ACCESS_TOKEN in environment. "
            "Options backtest is blocked."
        )
        return result

    # ── Case A: Refresh token present (preferred) ─────────────────────────────
    if rt:
        result["credential_source"]       = "REFRESH_TOKEN"
        result["token_refresh_attempted"] = True

        new_at, ref_http, ref_err = _exchange_refresh(rt)
        result["token_refresh_http_status"] = ref_http

        if not new_at:
            result["token_refresh_status"] = "FAILED"
            result["reason"]               = f"Refresh token exchange failed: {ref_err}"
            return result

        result["token_refresh_status"] = "SUCCESS"
        working_token = new_at

    else:
        # ── Case B: Only access token (may be stale) ──────────────────────────
        result["credential_source"]    = "ACCESS_TOKEN_ONLY"
        result["token_refresh_status"] = "NOT_RUN"
        result["reason"] = (
            "Only TASTYTRADE_ACCESS_TOKEN present — no refresh token. "
            "Checking if token is still valid via /customers/me."
        )
        working_token = at

    # ── Customer check (real API call) ────────────────────────────────────────
    result["customer_check_attempted"] = True
    cust_ok, cust_http, cust_err       = _check_customers_me(working_token)
    result["auth_http_status"]         = cust_http

    if cust_ok:
        result["customer_check_status"] = "SUCCESS"
        result["backtest_allowed"]      = True
        result["reason"] = (
            f"Auth verified via {result['credential_source']}. "
            f"Token refresh: {result['token_refresh_status']}. "
            "/customers/me returned HTTP 200. Backtest allowed."
        )
    else:
        result["customer_check_status"] = "FAILED"
        result["backtest_allowed"]      = False
        result["reason"]                = (
            f"Customer check failed ({cust_err}). "
            "Token may be expired. Backtest blocked."
        )

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    force = os.getenv("FORCE_DISABLE_TASTYTRADE", "false").lower() == "true"

    print("\n-- Tastytrade Auth Truth Verification --\n")
    if force:
        print("  MODE: FORCE_DISABLE_TASTYTRADE=true (verification mode)\n")
    else:
        rt_present = bool(os.getenv("TASTYTRADE_REFRESH_TOKEN", "").strip())
        at_present = bool(os.getenv("TASTYTRADE_ACCESS_TOKEN", "").strip())
        print(f"  TASTYTRADE_REFRESH_TOKEN present : {rt_present}")
        print(f"  TASTYTRADE_ACCESS_TOKEN present  : {at_present}")
        print(f"  API base : {_TT_BASE}")
        if rt_present:
            print("  Path     : REFRESH_TOKEN -> exchange -> /customers/me")
        elif at_present:
            print("  Path     : ACCESS_TOKEN_ONLY -> /customers/me (may be expired)")
        else:
            print("  Path     : MISSING -> blocked immediately")
        print()

    result = verify_tastytrade_auth_truth()
    print(json.dumps(result, indent=2))

    if result["backtest_allowed"]:
        print("\n  RESULT: backtest_allowed=true  -- Options backtest will proceed.")
        sys.exit(0)
    else:
        print(f"\n  RESULT: backtest_allowed=false -- {result['reason']}")
        sys.exit(1)
