"""Bulk-test option trading signals against the Discord agent option engine.

This runner intentionally reuses the same parser, strategy validation, Alpaca
contract lookup, queue storage, and paper order helpers used by discord_agent.py.
It can repeat a smaller JSON signal set until a requested sample count is reached.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .alpaca_paper import AlpacaPaperClient
from .discord_agent import (
    DecisionResult,
    _allowed_option_strategy_statuses,
    _as_float,
    _is_market_closed_order_error,
    _option_final_decision,
    _option_limit_entry_ready,
    _option_mapped_action,
    _option_order_route,
    _option_signal_quality,
    _queue_option_order,
    _record_order,
    _resolved_option_qty,
    _verify_exact_option_contract,
)
from .options_parser import classify_and_parse
from .options_strategy_bridge import _cache_key, _read_cache, build_options_strategy_input, run_options_strategy_validation
from .state_store import add_decision_history, record_option_journal_entry, upsert_option_position


def _load_signals(path: Path) -> List[str]:
    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return [chunk.strip() for chunk in raw.replace("\r\n", "\n").split("\n\n") if chunk.strip()]
    if isinstance(data, list):
        out: List[str] = []
        for item in data:
            if isinstance(item, str):
                out.append(item.strip())
            elif isinstance(item, dict) and item.get("signal"):
                out.append(str(item["signal"]).strip())
        return [s for s in out if s]
    if isinstance(data, dict):
        values = data.get("signals") or data.get("data") or []
        if isinstance(values, list):
            return [str(v.get("signal") if isinstance(v, dict) else v).strip() for v in values if v]
    return []


def _repeat_to_count(signals: List[str], count: int) -> List[str]:
    if not signals or count <= 0:
        return []
    return [signals[i % len(signals)] for i in range(count)]


def _status_counts(rows: Iterable[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "-")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _resolve_contract(alpaca: AlpacaPaperClient, option: Any, contract_check: Dict[str, Any]) -> tuple[str, str, str]:
    contracts = contract_check.get("contracts")
    lookup_err = contract_check.get("message", "")
    used_fallback = ""
    if not contracts and option.expiry_mode == "0dte":
        contracts, lookup_err = alpaca.get_option_contracts(option.root, None, option.strike, option.side.lower())
        if contracts:
            contracts = sorted(contracts, key=lambda c: str(c.get("expiration_date") or ""))
            used_fallback = "nearest_expiry"
    if not contracts:
        return "", "", lookup_err
    contract = contracts[0]
    return str(contract.get("symbol") or "").upper(), str(contract.get("expiration_date") or option.expiry_date or ""), used_fallback


def _queue_pending(option: Any, qty: float, order_type: str, expiration: str, quality: Dict[str, Any], reason: str, order_side: str, position_intent: str, requires_position: bool, occ_symbol: str = "") -> None:
    _queue_option_order(
        occ_symbol,
        option,
        qty,
        order_type,
        expiration or option.expiry_date or "",
        quality,
        reason,
        order_side,
        position_intent,
        requires_position,
        not bool(occ_symbol),
    )
    _record_order(occ_symbol or option.root, order_side, qty, "watching", "", "option", reason)


def _execute_or_queue_option(alpaca: AlpacaPaperClient, option: Any, quality: Dict[str, Any], max_submit_state: Dict[str, int]) -> Dict[str, Any]:
    order_side, position_intent, requires_position = _option_order_route(option)
    qty = _resolved_option_qty(option)
    order_type = "limit" if option.fill_price else "market"

    if not alpaca.ready():
        return {"paper_status": "SKIPPED", "paper_detail": "Alpaca paper trading is not configured or disabled.", "order_id": "", "occ_symbol": ""}

    options_enabled, opt_err = alpaca.has_options_trading()
    if not options_enabled:
        return {"paper_status": "SKIPPED", "paper_detail": opt_err, "order_id": "", "occ_symbol": ""}

    contract_check = _verify_exact_option_contract(option)
    occ_symbol, expiration, fallback = _resolve_contract(alpaca, option, contract_check)
    if not occ_symbol:
        _queue_pending(option, qty, order_type, option.expiry_date or "", quality, "waiting_for_tradable_contract", order_side, position_intent, requires_position)
        return {"paper_status": "QUEUED", "paper_detail": "Waiting for exact tradable Alpaca contract.", "order_id": "", "occ_symbol": ""}

    if requires_position:
        position, _ = alpaca.get_position(occ_symbol)
        held_qty = abs(_as_float((position or {}).get("qty")))
        if held_qty <= 0:
            _queue_pending(option, qty, "market", expiration, quality, "waiting_for_matching_position", order_side, position_intent, True, occ_symbol)
            return {"paper_status": "QUEUED", "paper_detail": "Waiting for matching Alpaca option position.", "order_id": "", "occ_symbol": occ_symbol}
        qty = min(qty, held_qty)

    open_order, _ = alpaca.has_open_order(occ_symbol)
    if open_order:
        _queue_pending(option, qty, order_type, expiration, quality, "existing_open_order", order_side, position_intent, requires_position, occ_symbol)
        return {"paper_status": "QUEUED", "paper_detail": "Existing open order; waiting.", "order_id": "", "occ_symbol": occ_symbol}

    market_open, market_err = alpaca.is_market_open()
    if not market_open:
        _queue_pending(option, qty, order_type, expiration, quality, market_err or "market_closed", order_side, position_intent, requires_position, occ_symbol)
        return {"paper_status": "QUEUED", "paper_detail": "Market closed; queued for open.", "order_id": "", "occ_symbol": occ_symbol}

    if order_type == "limit" and option.fill_price:
        current, _ = alpaca.get_latest_option_price(occ_symbol)
        if not _option_limit_entry_ready(_as_float(current), option.fill_price, order_side):
            _queue_pending(option, qty, order_type, expiration, quality, "waiting_for_signal_limit_price", order_side, position_intent, requires_position, occ_symbol)
            return {"paper_status": "QUEUED", "paper_detail": "Waiting for signal limit price.", "order_id": "", "occ_symbol": occ_symbol}

    if max_submit_state["submitted"] >= max_submit_state["max"]:
        _queue_pending(option, qty, order_type, expiration, quality, "bulk_submit_cap_reached", order_side, position_intent, requires_position, occ_symbol)
        return {"paper_status": "QUEUED", "paper_detail": "Bulk submit cap reached; queued safely.", "order_id": "", "occ_symbol": occ_symbol}

    order, err = alpaca.submit_option_order(occ_symbol, order_side, qty, order_type, option.fill_price, position_intent)
    if not order:
        if _is_market_closed_order_error(err):
            _queue_pending(option, qty, order_type, expiration, quality, err, order_side, position_intent, requires_position, occ_symbol)
            return {"paper_status": "QUEUED", "paper_detail": "Alpaca says market closed; queued.", "order_id": "", "occ_symbol": occ_symbol}
        return {"paper_status": "FAILED", "paper_detail": err, "order_id": "", "occ_symbol": occ_symbol}

    max_submit_state["submitted"] += 1
    order_id = str(order.get("id") or "")
    _record_order(occ_symbol, order_side, qty, "submitted", order_id, "option", "bulk_option_test")
    if position_intent == "buy_to_open":
        upsert_option_position(occ_symbol, option.root, option.side, option.strike, expiration, qty, option.fill_price or 0.0, order_id, option.stop_loss, option.target_price, quality.get("score"))
    record_option_journal_entry({
        "occ_symbol": occ_symbol,
        "root": option.root,
        "side": option.side,
        "strike": option.strike,
        "expiry_date": expiration,
        "quantity": qty,
        "order_type": order_type,
        "position_intent": position_intent,
        "raw_input": option.raw_text,
        "status": "bulk_order_submitted",
    })
    detail = f"{order_side.upper()} {qty:g}, intent={position_intent}, type={order_type}"
    if fallback:
        detail += f", {fallback}"
    return {"paper_status": str(order.get("status") or "SUBMITTED").upper(), "paper_detail": detail, "order_id": order_id, "occ_symbol": occ_symbol}


def _cached_validation_for(option: Any) -> Dict[str, Any] | None:
    strategy_input = build_options_strategy_input(option)
    cache = _read_cache()
    for candidate in (strategy_input, {**strategy_input, "validation_method": strategy_input.get("strike_selection")}):
        hit = cache.get(_cache_key(candidate)) or {}
        if hit.get("result"):
            result = dict(hit["result"])
            result["cache_hit"] = True
            return result
    return None


def run(input_path: Path, output_dir: Path, sample_count: int, execute_paper: bool, max_submit_orders: int, cache_only: bool = False) -> List[Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_signals = _load_signals(input_path)
    signals = _repeat_to_count(source_signals, sample_count)
    alpaca = AlpacaPaperClient()
    submit_state = {"submitted": 0, "max": max_submit_orders}
    validation_cache: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []

    for idx, signal in enumerate(signals, start=1):
        routed = classify_and_parse(signal)
        row: Dict[str, Any] = {
            "index": idx,
            "input": signal,
            "kind": routed.kind,
            "symbol": "",
            "action": "",
            "strike": "",
            "side": "",
            "expiry": "",
            "qty": "",
            "quality_score": "",
            "strategy_status": "",
            "strategy_decision": "",
            "final_output": "",
            "paper_status": "DRY_RUN" if not execute_paper else "SKIPPED",
            "paper_detail": "",
            "order_id": "",
            "occ_symbol": "",
            "reason": routed.reason,
        }

        if routed.kind != "OPTION" or not routed.option:
            row["final_output"] = routed.kind
            rows.append(row)
            continue

        option = routed.option
        row.update({
            "symbol": option.root,
            "action": option.order_action,
            "strike": option.strike if option.strike is not None else "",
            "side": option.side or "",
            "expiry": option.expiry_date or "",
            "qty": _resolved_option_qty(option) if option.valid else "",
        })

        if not option.valid:
            row["final_output"] = "INVALID_OPTION"
            row["reason"] = option.reason
            rows.append(row)
            continue
        if option.is_multi_leg:
            row["final_output"] = "MULTI_LEG_TRACKED"
            row["reason"] = option.reason
            rows.append(row)
            continue
        if option.tense in {"past", "unknown"}:
            row["final_output"] = "JOURNAL_ONLY"
            row["reason"] = f"tense={option.tense}"
            rows.append(row)
            continue
        if option.order_action == "manage":
            row["final_output"] = "MANAGEMENT_ONLY"
            row["reason"] = "management signal"
            rows.append(row)
            continue

        quality = _option_signal_quality(option)
        row["quality_score"] = round(float(quality.get("score") or 0), 2)
        if not quality.get("passed"):
            validation = {"status": "SKIPPED_OPTION_QUALITY", "decision": "REVIEW", "error": "quality failed"}
        else:
            validation_key = "|".join([
                option.root,
                option.side or "",
                str(option.strike if option.strike is not None else option.delta_target),
                option.expiry_date or "",
                str(_resolved_option_qty(option)),
                option.order_action,
            ])
            validation = validation_cache.get(validation_key)
            if validation is None:
                if cache_only:
                    validation = _cached_validation_for(option)
                    if validation is None:
                        validation = {"status": "VALIDATION_CACHE_MISS", "decision": "REVIEW", "error": "No cached validation available during cache-only bulk run."}
                else:
                    validation = run_options_strategy_validation(option)
                validation_cache[validation_key] = dict(validation)
        strategy_decision = str(validation.get("decision") or "REVIEW").upper()
        mapped_action = _option_mapped_action(option)
        decision: DecisionResult = _option_final_decision(validation, quality)
        if strategy_decision == "SELL" and mapped_action == "SELL" and str(validation.get("status") or "").upper() in _allowed_option_strategy_statuses() and quality.get("passed"):
            decision = DecisionResult("SELL", "Option sell-side signal approved for paper-order preparation.", 0.0, 0.0, 0.0, float(quality.get("score") or 0), "options_validation")

        row.update({
            "strategy_status": validation.get("status", ""),
            "strategy_decision": strategy_decision,
            "final_output": decision.action if decision.action == mapped_action and strategy_decision == mapped_action else "REJECT",
            "reason": decision.reason,
        })
        add_decision_history({
            "source": "bulk_option_paper_test",
            "symbol": option.root,
            "kind": "OPTION",
            "user_action": mapped_action,
            "final_action": row["final_output"],
            "score": row["quality_score"],
            "reason": row["reason"],
            "raw_input": signal,
        })

        if execute_paper and row["final_output"] in {"BUY", "SELL"}:
            row.update(_execute_or_queue_option(alpaca, option, quality, submit_state))
        rows.append(row)
        if idx % 50 == 0:
            print(f"Processed {idx}/{len(signals)} | final={_status_counts(rows, 'final_output')} | paper={_status_counts(rows, 'paper_status')}", flush=True)

    csv_path = output_dir / "bulk_option_paper_test_results.csv"
    md_path = output_dir / "bulk_option_paper_test_summary.md"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Bulk Option Paper Test Summary\n\n")
        f.write(f"Input file: `{input_path}`\n\n")
        f.write(f"Source signals: {len(source_signals)}\n\n")
        f.write(f"Signals tested: {len(rows)}\n\n")
        f.write(f"Paper execution enabled: {execute_paper}\n\n")
        f.write(f"Cache-only validation: {cache_only}\n\n")
        f.write(f"Paper submit cap: {max_submit_orders}\n\n")
        f.write(f"Paper orders submitted now: {submit_state['submitted']}\n\n")
        f.write(f"Unique strategy validations: {len(validation_cache)}\n\n")
        for title, key in (("Kind", "kind"), ("Final Output", "final_output"), ("Paper Status", "paper_status"), ("Strategy Status", "strategy_status")):
            f.write(f"## {title}\n\n")
            for name, count in _status_counts(rows, key).items():
                f.write(f"- {name}: {count}\n")
            f.write("\n")
        f.write("## First 30 Rows\n\n")
        f.write("| # | Signal | Final | Paper | Detail |\n")
        f.write("|---:|---|---|---|---|\n")
        for row in rows[:30]:
            signal_text = str(row["input"]).replace("|", "/")[:90]
            detail = str(row.get("paper_detail") or row.get("reason") or "").replace("|", "/")[:90]
            f.write(f"| {row['index']} | {signal_text} | {row['final_output']} | {row['paper_status']} | {detail} |\n")
    print(f"Signals tested: {len(rows)}")
    print(f"Paper orders submitted now: {submit_state['submitted']}")
    print(f"CSV: {csv_path}")
    print(f"Summary: {md_path}")
    print(f"Final outputs: {_status_counts(rows, 'final_output')}")
    print(f"Paper statuses: {_status_counts(rows, 'paper_status')}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("discord_stock_prediction_agent/training_reports"))
    parser.add_argument("--sample-count", type=int, default=1000)
    parser.add_argument("--execute-paper", action="store_true")
    parser.add_argument("--max-submit-orders", type=int, default=25)
    parser.add_argument("--cache-only", action="store_true", help="Do not call external validators; use cached option validation only.")
    args = parser.parse_args()
    run(args.input_path, args.output_dir, args.sample_count, args.execute_paper, args.max_submit_orders, args.cache_only)


if __name__ == "__main__":
    main()
