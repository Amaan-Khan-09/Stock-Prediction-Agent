from __future__ import annotations

from datetime import date
import re
from typing import Any


_DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")


def _number(value: str | None) -> float | int:
    parsed = float(str(value or "0").replace(",", ""))
    return int(parsed) if parsed.is_integer() else parsed


def _expiry(text: str, resolved: str | None) -> str:
    match = _DATE_RE.search(text)
    if match:
        month, day, year = (int(part) for part in match.groups())
        return f"{month:02d}/{day:02d}/{year:04d}"
    try:
        parsed = date.fromisoformat(str(resolved or ""))
        return parsed.strftime("%m/%d/%Y")
    except ValueError:
        return str(resolved or "")


def _base(text: str, option: Any) -> dict[str, Any]:
    action = {
        "open_long": "BUY_TO_OPEN",
        "open_short": "SELL_TO_OPEN",
        "close_long": "SELL_TO_CLOSE",
        "close_short": "BUY_TO_CLOSE",
        "manage": "MANAGE_LONG_POSITION",
    }.get(str(option.order_action), "REVIEW")
    upper = text.upper()
    if re.search(r"\bSTC\s+\d+(?:\.\d+)?%", upper):
        action = "SCALE_OUT"
    if re.search(r"\bBOUGHT\s+\d+.*\bAVG\b.*\bSELL\b", upper):
        action = "MANAGE_LONG_POSITION"
    return {
        "asset_type": "OPTION",
        "trade_structure": "SINGLE_LEG",
        "action": action,
        "symbol": option.root,
        "option_type": option.side,
        "strike": None if option.strike is None else _number(str(option.strike)),
        "expiration": _expiry(text, option.expiry_date),
    }


def _position_fields(contract: dict[str, Any]) -> None:
    action = contract["action"]
    if action in {"SELL_TO_CLOSE", "SCALE_OUT", "MANAGE_LONG_POSITION"}:
        contract["position_requirement"] = "LONG_OPTION_POSITION"
        contract["status"] = "VALID_IF_POSITION_EXISTS"
    elif action == "BUY_TO_CLOSE":
        contract["position_requirement"] = "SHORT_OPTION_POSITION"
        contract["status"] = "VALID_IF_POSITION_EXISTS"
    else:
        contract["status"] = "VALID"


def _complex_contract(text: str, option: Any) -> dict[str, Any] | None:
    upper = text.upper()
    contract = _base(text, option)

    fills = re.search(
        r"\bGRABBED\s+(\d+)\s+[A-Z.]+\s+\d+(?:\.\d+)?[CP].*?@(\d+(?:\.\d+)?)"
        r".*?\bADDED\s+(\d+)\s+@(\d+(?:\.\d+)?)"
        r".*?\bTP\s+(\d+(?:\.\d+)?)",
        upper,
    )
    if fills:
        contract.update({
            "action": "BUY_TO_OPEN",
            "fills": [
                {"quantity": _number(fills.group(1)), "price": _number(fills.group(2))},
                {"quantity": _number(fills.group(3)), "price": _number(fills.group(4))},
            ],
            "average_entry": "CALCULATE_WEIGHTED",
            "take_profit": _number(fills.group(5)),
            "post_target_stop": "WEIGHTED_BREAKEVEN",
            "status": "VALID",
        })
        return contract

    ladder = re.search(
        r"\bSTARTER\s+[A-Z.]+\s+\d+(?:\.\d+)?[CP].*?@(\d+(?:\.\d+)?)"
        r".*?\bADD\s+(\d+)\s+@(\d+(?:\.\d+)?)\s+AND\s+(\d+)\s+@(\d+(?:\.\d+)?)"
        r".*?\bMAX\s+(\d+)\s+CONTRACTS?.*?\bSTOP\s+ENTIRE\s+POSITION\s+@(\d+(?:\.\d+)?)",
        upper,
    )
    if ladder:
        contract.update({
            "action": "BUY_TO_OPEN",
            "position_size": "STARTER",
            "initial_entry_price": _number(ladder.group(1)),
            "scale_in": [
                {"quantity": _number(ladder.group(2)), "limit_price": _number(ladder.group(3))},
                {"quantity": _number(ladder.group(4)), "limit_price": _number(ladder.group(5))},
            ],
            "maximum_contracts": _number(ladder.group(6)),
            "stop_loss": _number(ladder.group(7)),
            "stop_scope": "ENTIRE_POSITION",
            "status": "VALID",
        })
        return contract

    scale_out = re.search(
        r"\bSTC\s+(\d+(?:\.\d+)?)%\s+[A-Z.]+\s+\d+(?:\.\d+)?[CP].*?AT\s+(\d+(?:\.\d+)?)"
        r"\s*,\s*(\d+(?:\.\d+)?)%\s+AT\s+(\d+(?:\.\d+)?)"
        r".*?TRAIL\s+REST\s+(\d+(?:\.\d+)?)%",
        upper,
    )
    if scale_out:
        first_pct = _number(scale_out.group(1))
        second_pct = _number(scale_out.group(3))
        contract.update({
            "action": "SCALE_OUT",
            "exits": [
                {"percentage": first_pct, "order_type": "LIMIT", "limit_price": _number(scale_out.group(2))},
                {"percentage": second_pct, "order_type": "LIMIT", "limit_price": _number(scale_out.group(4))},
            ],
            "remaining_percentage": _number(str(100 - float(first_pct) - float(second_pct))),
            "remaining_trailing_stop_percent": _number(scale_out.group(5)),
            "event_exit": "BEFORE_EARNINGS",
            "position_requirement": "LONG_OPTION_POSITION",
            "status": "VALID_IF_POSITION_EXISTS",
        })
        return contract

    short_manage = re.search(
        r"\bBTC\s+HALF\s+[A-Z.]+\s+\d+(?:\.\d+)?[CP].*?@(\d+(?:\.\d+)?)"
        r".*?CLOSE\s+REST\s+AT\s+(\d+(?:\.\d+)?)%\s+PROFIT",
        upper,
    )
    if short_manage:
        contract.update({
            "action": "BUY_TO_CLOSE",
            "close_percentage": 50,
            "order_type": "LIMIT",
            "limit_price": _number(short_manage.group(1)),
            "remaining_position": {
                "stop_price": "ENTRY_CREDIT",
                "profit_target_percent": _number(short_manage.group(2)),
            },
            "position_requirement": "SHORT_OPTION_POSITION",
            "status": "VALID_IF_POSITION_EXISTS",
        })
        return contract

    conditional = re.search(
        r"\bBTO\s+[A-Z.]+\s+\d+(?:\.\d+)?[CP].*?MAX\s+PREMIUM\s+(\d+(?:\.\d+)?)"
        r"\s+ONLY\s+IF\s+[A-Z.]+\s*([<>])\s*(\d+(?:\.\d+)?)"
        r".*?CANCEL\s+(\d{1,2}):(\d{2})\s*PM\s*ET"
        r".*?MAX\s+RISK\s+\$(\d[\d,]*(?:\.\d+)?)"
        r".*?EXIT\s+(\d+)\s+MIN\s+BEFORE\s+CLOSE",
        upper,
    )
    if conditional:
        hour = int(conditional.group(4)) % 12 + 12
        contract.update({
            "action": "BUY_TO_OPEN",
            "order_type": "LIMIT",
            "maximum_entry_price": _number(conditional.group(1)),
            "entry_condition": {
                "reference": "UNDERLYING_PRICE",
                "operator": conditional.group(2),
                "value": _number(conditional.group(3)),
            },
            "cancel_if_not_triggered_by": {
                "time": f"{hour:02d}:{int(conditional.group(5)):02d}",
                "timezone": "ET",
            },
            "maximum_loss": _number(conditional.group(6)),
            "currency": "USD",
            "time_exit": {"offset_minutes_before_market_close": _number(conditional.group(7))},
            "status": "CONDITIONAL",
        })
        return contract

    managed = re.search(
        r"\bBOUGHT\s+(\d+)\s+[A-Z.]+\s+\d+(?:\.\d+)?[CP].*?AVG\s+(\d+(?:\.\d+)?)"
        r".*?SELL\s+(\d+)\s+AT\s+(\d+(?:\.\d+)?)"
        r"\s*,\s*SELL\s+(\d+)\s+AT\s+(\d+(?:\.\d+)?)",
        upper,
    )
    if managed:
        contract.update({
            "action": "MANAGE_LONG_POSITION",
            "current_quantity": _number(managed.group(1)),
            "average_entry": _number(managed.group(2)),
            "planned_exits": [
                {"quantity": _number(managed.group(3)), "limit_price": _number(managed.group(4))},
                {"quantity": _number(managed.group(5)), "limit_price": _number(managed.group(6))},
            ],
            "stop_adjustment": {
                "trigger": "FIRST_TAKE_PROFIT_FILL",
                "new_stop": "AVERAGE_ENTRY",
            },
            "emergency_exit": {
                "condition": "UNDERLYING_TRADING_HALT",
                "action": "CLOSE_ALL",
            },
            "position_requirement": "LONG_OPTION_POSITION",
            "status": "VALID_IF_POSITION_EXISTS",
        })
        return contract
    return None


def build_single_leg_contract(text: str, option: Any) -> dict[str, Any]:
    """Return the stable, user-facing execution contract for a single option leg."""
    complex_contract = _complex_contract(text, option)
    if complex_contract is not None:
        qty_match = re.search(r"\bQTY\s+(\d+(?:\.\d+)?)", text, re.IGNORECASE)
        if qty_match and not any(
            key in complex_contract
            for key in ("quantity", "initial_quantity", "current_quantity", "fills")
        ):
            complex_contract["quantity"] = _number(qty_match.group(1))
        return complex_contract

    upper = text.upper()
    contract = _base(text, option)
    action = contract["action"]

    partial = re.search(r"\bSTC\s+(\d+)\s+OF\s+(\d+)\b", upper)
    if partial:
        close_qty, total_qty = (_number(value) for value in partial.groups())
        contract.update({
            "close_quantity": close_qty,
            "total_position_quantity": total_qty,
            "remaining_quantity": _number(str(float(total_qty) - float(close_qty))),
        })
    elif re.search(r"\b(?:STC|BTC)\s+HALF\b", upper):
        contract["close_percentage"] = 50
    elif re.search(r"\bBTC\s+ALL\b", upper):
        contract["quantity_scope"] = "ALL"
    elif re.search(r"\bQTY\s+(\d+(?:\.\d+)?)", upper):
        contract["quantity"] = _number(re.search(r"\bQTY\s+(\d+(?:\.\d+)?)", upper).group(1))

    market = bool(re.search(r"\b(?:AT\s+)?MARKET\b", upper))
    max_price = re.search(r"\bMAX(?:IMUM)?(?:\s+PREMIUM)?\s+(\d+(?:\.\d+)?)", upper)
    explicit_limit = re.search(r"\bLIMIT\s+(\d+(?:\.\d+)?)", upper)
    at_price = re.search(r"@(\d+(?:\.\d+)?)", upper)
    plain_at = None
    for candidate in re.finditer(r"\bAT\s+(\d+(?:\.\d+)?)", upper):
        prefix = upper[max(0, candidate.start() - 48):candidate.start()]
        if re.search(
            r"\b(?:SL|STOP|STOPLOSS|TARGET|TGT|TP|PT|PROFIT|TRIM|SELL|CLOSE)\b",
            prefix,
        ):
            continue
        plain_at = candidate
        break
    if market:
        contract["order_type"] = "MARKET"
    elif max_price:
        contract["order_type"] = "LIMIT"
        contract["maximum_entry_price"] = _number(max_price.group(1))
    elif explicit_limit or at_price or plain_at:
        price_match = explicit_limit or at_price or plain_at
        contract["order_type"] = "LIMIT"
        contract["limit_price"] = _number(price_match.group(1))
    elif action in {"BUY_TO_OPEN", "SELL_TO_OPEN", "SELL_TO_CLOSE", "BUY_TO_CLOSE"}:
        contract["order_type"] = "MARKET"

    qty_match = re.search(r"\bQTY\s+(\d+(?:\.\d+)?)", upper)
    if qty_match:
        contract["quantity"] = _number(qty_match.group(1))

    tif_match = re.search(r"\b(GTC|DAY|IOC|FOK)\b", upper)
    if tif_match:
        contract["time_in_force"] = tif_match.group(1)

    targets = [_number(value) for value in re.findall(r"\bTP\d*\s+(\d+(?:\.\d+)?)", upper)]
    if targets:
        contract["take_profit"] = targets if len(targets) > 1 else targets[0]
    stop = re.search(r"\bSL\s+(\d+(?:\.\d+)?)", upper)
    if stop:
        contract["stop_loss"] = _number(stop.group(1))

    trail = re.search(r"\bTRAIL(?:ING)?(?:\s+THE\s+FINAL\s+CONTRACT\s+BY|\s+STOP)?\s+(\d+(?:\.\d+)?)%", upper)
    if trail:
        key = "remaining_trailing_stop_percent" if partial else "trailing_stop_percent"
        contract[key] = _number(trail.group(1))

    planned_btc = re.search(r"\bBTC\s+AT\s+(\d+(?:\.\d+)?)", upper)
    premium_stop = re.search(r"\bSTOP\s+IF\s+PREMIUM\s+(?:HITS|REACHES)\s+(\d+(?:\.\d+)?)", upper)
    if planned_btc:
        if premium_stop:
            contract["planned_exit"] = {
                "take_profit_buy_to_close": _number(planned_btc.group(1)),
                "stop_price": _number(premium_stop.group(1)),
            }
        else:
            contract["planned_exit"] = {
                "action": "BUY_TO_CLOSE",
                "limit_price": _number(planned_btc.group(1)),
            }

    timed = re.search(r"\bEXIT\s+(?:(\d+)\s+MINUTES?\s+)?BEFORE\s+CLOSE", upper)
    if timed:
        if timed.group(1):
            value: str | dict[str, Any] = {
                "offset_minutes_before_market_close": _number(timed.group(1))
            }
            if "IF TP NOT HIT" in upper:
                value["condition"] = "TAKE_PROFIT_NOT_HIT"
            contract["time_exit"] = value
        else:
            contract["time_exit"] = "BEFORE_MARKET_CLOSE"

    starter = re.search(
        r"\bSTARTER\s+[A-Z.]+\s+\d+(?:\.\d+)?[CP].*?@(\d+(?:\.\d+)?)"
        r".*?ADD\s+(\d+)\s+CONTRACTS?\s+IF\s+[A-Z.]+\s+BREAKS\s+(ABOVE|BELOW)\s+(\d+(?:\.\d+)?)"
        r".*?RISK\s+(\d+(?:\.\d+)?)%",
        upper,
    )
    if starter:
        contract.pop("quantity", None)
        contract.update({
            "position_size": "STARTER",
            "order_type": "LIMIT",
            "limit_price": _number(starter.group(1)),
            "scale_in": {
                "quantity": _number(starter.group(2)),
                "condition": {
                    "reference": "UNDERLYING_PRICE",
                    "operator": ">" if starter.group(3) == "ABOVE" else "<",
                    "value": _number(starter.group(4)),
                },
            },
            "portfolio_risk_percent": _number(starter.group(5)),
            "status": "CONDITIONAL",
        })
        return contract

    conditional = re.search(
        r"\bONLY\s+IF\s+[A-Z.]+\s+(?:TRADES\s+)?(?:IS\s+)?([<>]|ABOVE|BELOW)\s*(\d+(?:\.\d+)?)",
        upper,
    )
    if conditional:
        operator = conditional.group(1)
        operator = ">" if operator == "ABOVE" else "<" if operator == "BELOW" else operator
        contract["entry_condition"] = {
            "reference": "UNDERLYING_PRICE",
            "operator": operator,
            "value": _number(conditional.group(2)),
        }
        cancel = re.search(r"\bCANCEL(?:\s+AT)?\s+(\d{1,2}):(\d{2})\s*PM\s*ET", upper)
        if cancel:
            hour = int(cancel.group(1)) % 12 + 12
            contract["cancel_if_not_triggered_by"] = {
                "time": f"{hour:02d}:{int(cancel.group(2)):02d}",
                "timezone": "ET",
            }
        contract["status"] = "CONDITIONAL"

    _position_fields(contract)
    if conditional or starter:
        contract["status"] = "CONDITIONAL"
    return contract
