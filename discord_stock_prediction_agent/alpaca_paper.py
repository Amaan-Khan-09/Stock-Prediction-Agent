"""Minimal Alpaca paper trading client used by the Discord agent."""
from __future__ import annotations

import time
import threading
from math import gcd
from typing import Any, Dict, Optional, Tuple
from urllib.parse import quote

import requests

from .config import config


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        number = float(value)
        if number <= 0:
            return None
        return number
    except (TypeError, ValueError):
        return None


def _snapshot_for_symbol(data: Dict[str, Any], occ_symbol: str) -> Dict[str, Any]:
    snapshots = (data or {}).get("snapshots") or {}
    symbol = str(occ_symbol or "").upper()
    if isinstance(snapshots, dict):
        return snapshots.get(symbol) or snapshots.get(occ_symbol) or {}
    if isinstance(snapshots, list):
        for item in snapshots:
            if str(item.get("symbol") or item.get("S") or "").upper() == symbol:
                return item
    return {}


def _extract_option_snapshot_price(data: Dict[str, Any], occ_symbol: str) -> Optional[float]:
    snapshot = _snapshot_for_symbol(data, occ_symbol)
    if not snapshot:
        return None

    quote = snapshot.get("latestQuote") or snapshot.get("latest_quote") or {}
    bid = _safe_float(quote.get("bp") or quote.get("bid_price"))
    ask = _safe_float(quote.get("ap") or quote.get("ask_price"))
    if bid is not None and ask is not None:
        return round((bid + ask) / 2, 6)
    if bid is not None:
        return bid
    if ask is not None:
        return ask

    trade = snapshot.get("latestTrade") or snapshot.get("latest_trade") or {}
    for key in ("p", "price"):
        price = _safe_float(trade.get(key))
        if price is not None:
            return price

    minute_bar = snapshot.get("minuteBar") or snapshot.get("minute_bar") or {}
    for key in ("c", "close"):
        price = _safe_float(minute_bar.get(key))
        if price is not None:
            return price
    return None


def _extract_option_snapshot_quote(data: Dict[str, Any], occ_symbol: str) -> Optional[Dict[str, float]]:
    """Like _extract_option_snapshot_price, but keeps bid and ask
    separate instead of collapsing straight to a mid-price -- callers
    that need to judge liquidity (a wide bid/ask spread on an illiquid
    contract) need the two numbers, not just their midpoint. Alpaca's
    snapshot response already carries both; this doesn't add a new
    network call.
    """
    snapshot = _snapshot_for_symbol(data, occ_symbol)
    if not snapshot:
        return None
    quote = snapshot.get("latestQuote") or snapshot.get("latest_quote") or {}
    bid = _safe_float(quote.get("bp") or quote.get("bid_price"))
    ask = _safe_float(quote.get("ap") or quote.get("ask_price"))
    if bid is None and ask is None:
        return None
    mid = round((bid + ask) / 2, 6) if bid is not None and ask is not None else (bid or ask)
    return {"bid": bid or 0.0, "ask": ask or 0.0, "mid": mid}


class AlpacaPaperClient:
    def __init__(self) -> None:
        self.base_url = config.alpaca_base_url
        self.data_base_url = config.alpaca_data_base_url
        self._request_slots = threading.BoundedSemaphore(
            max(1, min(32, config.alpaca_max_concurrent_requests))
        )

    def _headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": config.alpaca_api_key,
            "APCA-API-SECRET-KEY": config.alpaca_secret_key,
            "Content-Type": "application/json",
        }

    def ready(self) -> bool:
        return config.paper_trading_enabled and config.has_alpaca

    def get_account(self) -> Tuple[Optional[Dict[str, Any]], str]:
        return self._get("/v2/account")

    def get_clock(self) -> Tuple[Optional[Dict[str, Any]], str]:
        return self._get("/v2/clock")

    def is_market_open(self) -> Tuple[bool, str]:
        clock, err = self.get_clock()
        if not clock:
            return False, err or "Alpaca market clock unavailable."
        return bool(clock.get("is_open")), ""

    def get_position(self, symbol: str) -> Tuple[Optional[Dict[str, Any]], str]:
        data, err = self._get(f"/v2/positions/{symbol.upper()}")
        if err and "404" in err:
            return None, "No position found."
        return data, err

    def get_order(self, order_id: str) -> Tuple[Optional[Dict[str, Any]], str]:
        return self._get(f"/v2/orders/{order_id}")

    def get_order_by_client_order_id(
        self, client_order_id: str
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        if not client_order_id:
            return None, "client_order_id is required."
        encoded = quote(str(client_order_id), safe="")
        return self._get(f"/v2/orders:by_client_order_id?client_order_id={encoded}")

    def get_latest_price(self, symbol: str) -> Tuple[Optional[float], str]:
        position, _ = self.get_position(symbol)
        if position:
            for key in ("current_price", "market_value"):
                value = position.get(key)
                try:
                    if key == "current_price" and value is not None:
                        return float(value), ""
                except (TypeError, ValueError):
                    pass
        data, err = self._get_data(f"/v2/stocks/{symbol.upper()}/trades/latest")
        if data:
            trade = data.get("trade") or {}
            price = trade.get("p")
            try:
                return float(price), ""
            except (TypeError, ValueError):
                pass
        return None, err or "Latest price unavailable."


    def get_latest_option_market_price(self, occ_symbol: str) -> Tuple[Optional[float], str]:
        """Return a market-data premium for an OCC option contract.

        This intentionally ignores any held position price. Entry-limit checks
        must use a fresh option quote/trade snapshot, otherwise an old position
        value can make a new signal look wrongly reachable or unreachable.
        """
        symbol = occ_symbol.upper()
        data, err = self._get_data(f"/v1beta1/options/snapshots?symbols={symbol}")
        if data:
            price = _extract_option_snapshot_price(data, symbol)
            if price is not None:
                return price, ""
        return None, err or "Latest option market premium unavailable."

    def get_option_quote(self, occ_symbol: str) -> Tuple[Optional[Dict[str, float]], str]:
        """Return {"bid", "ask", "mid"} for an OCC option contract symbol,
        so callers can judge liquidity (spread width), not just get a
        single collapsed price. Same underlying snapshot endpoint as
        get_latest_option_market_price -- this doesn't cost an extra
        request when both are needed for the same symbol.
        """
        symbol = occ_symbol.upper()
        data, err = self._get_data(f"/v1beta1/options/snapshots?symbols={symbol}")
        if data:
            quote = _extract_option_snapshot_quote(data, symbol)
            if quote is not None:
                return quote, ""
        return None, err or "Latest option quote unavailable."

    def get_latest_option_price(self, occ_symbol: str) -> Tuple[Optional[float], str]:
        """Return the latest option premium for an OCC contract symbol.

        For SL/TP monitoring on options, prefer the option position's
        current_price when available, then fall back to the option market-data
        snapshot. New entry-limit checks should call get_latest_option_market_price().
        """
        symbol = occ_symbol.upper()
        position, _ = self.get_position(symbol)
        if position:
            price = _safe_float(position.get("current_price"))
            if price is not None:
                return price, ""

        return self.get_latest_option_market_price(symbol)

    def submit_market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        client_order_id: str = "",
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        return self.submit_equity_order(
            symbol=symbol,
            side=side,
            qty=qty,
            order_type="market",
            time_in_force="day",
            client_order_id=client_order_id,
        )

    def submit_equity_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "day",
        client_order_id: str = "",
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """Submit an equity market, limit, stop, or stop-limit paper order."""
        if not self.ready():
            return None, "Alpaca paper trading is not configured or disabled."
        if qty <= 0:
            return None, "Quantity must be greater than zero."
        normalized_type = str(order_type or "market").lower()
        if normalized_type not in {"market", "limit", "stop", "stop_limit"}:
            return None, f"Unsupported equity order type: {normalized_type}."
        tif = str(time_in_force or "day").lower()
        if tif not in {"day", "gtc", "ioc", "fok"}:
            return None, f"Unsupported time in force: {tif}."
        payload: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "qty": str(qty),
            "side": side.lower(),
            "type": normalized_type,
            "time_in_force": tif,
        }
        if normalized_type in {"limit", "stop_limit"}:
            if limit_price is None or float(limit_price) <= 0:
                return None, "limit_price is required for this equity order."
            payload["limit_price"] = str(limit_price)
        if normalized_type in {"stop", "stop_limit"}:
            if stop_price is None or float(stop_price) <= 0:
                return None, "stop_price is required for this equity order."
            payload["stop_price"] = str(stop_price)
        if client_order_id:
            payload["client_order_id"] = str(client_order_id)[:48]
        return self._post("/v2/orders", payload)

    def wait_for_order(self, order_id: str, seconds: int = 10) -> Tuple[Optional[Dict[str, Any]], str]:
        last_data: Optional[Dict[str, Any]] = None
        last_err = ""
        for _ in range(max(1, seconds)):
            data, err = self._get(f"/v2/orders/{order_id}")
            if data:
                last_data = data
                if str(data.get("status", "")).lower() in {
                    "filled", "partially_filled", "canceled", "expired", "rejected"
                }:
                    return data, ""
            last_err = err
            time.sleep(1)
        return last_data, last_err

    def has_open_order(self, symbol: str) -> Tuple[bool, str]:
        data, err = self._get(f"/v2/orders?status=open&symbols={symbol.upper()}")
        if err:
            return False, err
        if isinstance(data, list):
            return len(data) > 0, ""
        return False, ""

    def has_sellable_quantity(self, symbol: str, qty: float) -> Tuple[bool, float, str]:
        position, err = self.get_position(symbol)
        if not position:
            return False, 0.0, err or "No position found."
        try:
            held = float(position.get("qty") or 0)
        except (TypeError, ValueError):
            held = 0.0
        if held <= 0:
            return False, held, "No long quantity is available to sell."
        if qty > held:
            return False, held, f"Requested qty {qty:g}, but Alpaca position has {held:g}."
        return True, held, ""

    # ── Options ──────────────────────────────────────────────────────────────

    def has_options_trading(self) -> Tuple[bool, str]:
        """Check whether this Alpaca account has options trading enabled.

        Must be enabled from the Alpaca dashboard -- there is no API call that
        turns it on. This only reports current status.
        """
        account, err = self.get_account()
        if not account:
            return False, err or "Could not read Alpaca account status."
        level = account.get("options_trading_level")
        approved_at = account.get("options_approved_at")
        enabled = bool(approved_at) or (level is not None and int(level or 0) > 0)
        if enabled:
            return True, ""
        return False, (
            "Options trading is not enabled on this Alpaca account. "
            "Enable it in the Alpaca dashboard before options orders can be placed."
        )

    def has_multi_leg_options_trading(self) -> Tuple[bool, str]:
        """Return whether Alpaca reports the Level 3 permission required for MLeg orders."""
        account, err = self.get_account()
        if not account:
            return False, err or "Could not read Alpaca account status."
        try:
            level = int(account.get("options_trading_level") or 0)
        except (TypeError, ValueError):
            level = 0
        if level >= 3:
            return True, ""
        return False, (
            f"Alpaca Options Level 3 is required for multi-leg orders; current level is {level}."
        )

    def get_option_contracts(
        self,
        underlying: str,
        expiration_date: Optional[str] = None,
        strike: Optional[float] = None,
        option_type: Optional[str] = None,
    ) -> Tuple[Optional[list], str]:
        """Look up real, tradable option contracts for an underlying symbol.

        Always call this before submit_option_order() -- it confirms the
        constructed OCC symbol (or nearest available expiry) actually exists
        rather than guessing and letting the order call fail.
        """
        params = [f"underlying_symbols={underlying.upper()}"]
        if expiration_date:
            params.append(f"expiration_date={expiration_date}")
        if strike is not None:
            params.append(f"strike_price_gte={strike}")
            params.append(f"strike_price_lte={strike}")
        if option_type:
            params.append(f"type={option_type.lower()}")
        query = "&".join(params)
        data, err = self._get(f"/v2/options/contracts?{query}")
        if err:
            return None, err
        contracts = (data or {}).get("option_contracts")
        if not contracts:
            return None, (
                f"No tradable option contract found for {underlying.upper()} "
                f"(expiration={expiration_date or 'any'}, strike={strike}, type={option_type})."
            )
        return contracts, ""

    def submit_option_order(
        self,
        occ_symbol: str,
        side: str,
        qty: float,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        position_intent: Optional[str] = None,
        client_order_id: str = "",
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """Submit a single-leg options order. side is 'buy' or 'sell'.

        Options orders on Alpaca must use time_in_force='day'. position_intent
        may be buy_to_open, buy_to_close, sell_to_open, or sell_to_close.
        """
        if not self.ready():
            return None, "Alpaca paper trading is not configured or disabled."
        if qty <= 0:
            return None, "Quantity must be greater than zero."
        contracts_qty = int(qty)
        if contracts_qty <= 0 or abs(float(qty) - contracts_qty) > 1e-9:
            return None, "Options quantity must be a whole number of contracts."
        payload: Dict[str, Any] = {
            "symbol": occ_symbol,
            "qty": str(contracts_qty),
            "side": side.lower(),
            "type": order_type.lower(),
            "time_in_force": "day",
        }
        if position_intent:
            payload["position_intent"] = position_intent
        if client_order_id:
            payload["client_order_id"] = str(client_order_id)[:48]
        if order_type.lower() == "limit":
            if limit_price is None:
                return None, "limit_price is required for a limit order."
            payload["limit_price"] = str(limit_price)
        return self._post("/v2/orders", payload)

    def submit_multi_leg_option_order(
        self,
        legs: list[Dict[str, Any]],
        qty: float,
        order_type: str = "limit",
        limit_price: Optional[float] = None,
        client_order_id: str = "",
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """Submit an Alpaca multi-leg option order.

        Alpaca requires order_class='mleg' and 2-4 option legs. Each leg must
        include symbol, side, ratio_qty, and position_intent. The caller is
        responsible for contract lookup and strategy validation before this.
        """
        if not self.ready():
            return None, "Alpaca paper trading is not configured or disabled."
        if qty <= 0:
            return None, "Quantity must be greater than zero."
        contracts_qty = int(qty)
        if contracts_qty <= 0 or abs(float(qty) - contracts_qty) > 1e-9:
            return None, "Options quantity must be a whole number of contracts."
        if not (2 <= len(legs or []) <= 4):
            return None, "Multi-leg option orders require 2 to 4 legs."
        allowed_intents = {"buy_to_open", "buy_to_close", "sell_to_open", "sell_to_close"}
        normalized_legs = []
        ratios = []
        for index, leg in enumerate(legs or [], start=1):
            symbol = str(leg.get("symbol") or "").upper()
            side = str(leg.get("side") or "").lower()
            intent = str(leg.get("position_intent") or "").lower()
            try:
                ratio_qty = int(leg.get("ratio_qty") or 0)
            except (TypeError, ValueError):
                ratio_qty = 0
            if not symbol or side not in {"buy", "sell"} or intent not in allowed_intents or ratio_qty <= 0:
                return None, f"Multi-leg option leg {index} is incomplete or invalid."
            ratios.append(ratio_qty)
            normalized_legs.append(
                {
                    "symbol": symbol,
                    "ratio_qty": str(ratio_qty),
                    "side": side,
                    "position_intent": intent,
                }
            )
        common_ratio = ratios[0]
        for ratio in ratios[1:]:
            common_ratio = gcd(common_ratio, ratio)
        if common_ratio != 1:
            return None, "Multi-leg ratios must be reduced to their simplest form."
        payload: Dict[str, Any] = {
            "order_class": "mleg",
            "qty": str(contracts_qty),
            "type": order_type.lower(),
            "time_in_force": "day",
            "legs": normalized_legs,
        }
        if client_order_id:
            payload["client_order_id"] = str(client_order_id)[:48]
        if order_type.lower() == "limit":
            if limit_price is None:
                return None, "limit_price is required for a multi-leg limit order."
            payload["limit_price"] = str(limit_price)
        return self._post("/v2/orders", payload)

    def _get(self, path: str) -> Tuple[Optional[Dict[str, Any]], str]:
        return self._request("GET", f"{self.base_url}{path}")

    def _get_data(self, path: str) -> Tuple[Optional[Dict[str, Any]], str]:
        return self._request("GET", f"{self.data_base_url}{path}", data_api=True)

    def _post(self, path: str, payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
        data, error = self._request(
            "POST",
            f"{self.base_url}{path}",
            payload=payload,
            idempotent=bool(payload.get("client_order_id")),
        )
        if data or not payload.get("client_order_id"):
            return data, error
        existing, lookup_error = self.get_order_by_client_order_id(
            str(payload["client_order_id"])
        )
        if existing:
            return existing, ""
        return None, error or lookup_error

    def _request(
        self,
        method: str,
        url: str,
        payload: Optional[Dict[str, Any]] = None,
        data_api: bool = False,
        idempotent: bool = True,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        attempts = max(1, min(5, config.alpaca_max_request_attempts))
        timeout = max(5, min(60, config.alpaca_request_timeout_seconds))
        last_error = ""
        for attempt in range(1, attempts + 1):
            try:
                with self._request_slots:
                    response = requests.request(
                        method,
                        url,
                        headers=self._headers(),
                        json=payload,
                        timeout=timeout,
                    )
                if 200 <= response.status_code < 300:
                    return response.json(), ""
                prefix = "Alpaca data" if data_api else "Alpaca"
                last_error = (
                    f"{prefix} HTTP {response.status_code}: "
                    f"{response.text[:300 if method == 'POST' else 200]}"
                )
                retryable = response.status_code == 429 or response.status_code >= 500
                if not retryable or not idempotent or attempt >= attempts:
                    return None, last_error
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else min(2 ** (attempt - 1), 8)
            except Exception as exc:
                last_error = f"Alpaca request error: {type(exc).__name__}: {exc}"
                if not idempotent or attempt >= attempts:
                    return None, last_error
                delay = min(2 ** (attempt - 1), 8)
            time.sleep(max(0.1, min(float(delay), 15.0)))
        return None, last_error
