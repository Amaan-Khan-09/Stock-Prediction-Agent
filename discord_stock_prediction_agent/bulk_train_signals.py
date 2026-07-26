"""Run many text signals through the Discord agent decision engine.

By default this is a dry run and never places Alpaca orders. With
--execute-paper, it places Alpaca paper trades only for applicable final BUY/SELL
outputs and records execution status in the reports.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

from .alpaca_paper import AlpacaPaperClient
from .discord_agent import _evaluate_final_action, _resolved_option_qty
from .options_parser import classify_and_parse
from .options_symbol import resolve_underlying_for_prediction
from .prediction_bridge import run_project_prediction
from .state_store import add_decision_history


def _split_signals(text: str) -> List[str]:
    return [chunk.strip() for chunk in text.replace("\r\n", "\n").split("\n\n") if chunk.strip()]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _execute_paper_trade(alpaca: AlpacaPaperClient, action: str, symbol: str, quantity: float | None) -> Dict[str, Any]:
    if not alpaca.ready():
        return {"status": "SKIPPED", "detail": "Alpaca paper trading is not configured or disabled.", "order_id": ""}

    open_order, open_err = alpaca.has_open_order(symbol)
    if open_order:
        return {"status": "SKIPPED", "detail": "Existing open Alpaca order for this symbol.", "order_id": ""}
    if open_err:
        return {"status": "WARNING", "detail": f"Open-order check warning: {open_err}", "order_id": ""}

    if action == "BUY":
        qty = quantity or 1.0
        order, err = alpaca.submit_market_order(symbol, "buy", qty)
        if not order:
            return {"status": "FAILED", "detail": err, "order_id": ""}
        return {"status": str(order.get("status") or "submitted").upper(), "detail": f"BUY qty {qty:g}", "order_id": str(order.get("id") or "")}

    if action == "SELL":
        if quantity is None:
            position, pos_err = alpaca.get_position(symbol)
            if not position:
                return {"status": "SKIPPED", "detail": pos_err or "No Alpaca position found.", "order_id": ""}
            qty = _as_float(position.get("qty"))
        else:
            qty = quantity
        ok, held, reason = alpaca.has_sellable_quantity(symbol, qty)
        if not ok:
            return {"status": "SKIPPED", "detail": reason, "order_id": ""}
        order, err = alpaca.submit_market_order(symbol, "sell", min(qty, held))
        if not order:
            return {"status": "FAILED", "detail": err, "order_id": ""}
        return {"status": str(order.get("status") or "submitted").upper(), "detail": f"SELL qty {min(qty, held):g}", "order_id": str(order.get("id") or "")}

    return {"status": "SKIPPED", "detail": f"No paper trade for final output {action}.", "order_id": ""}


def run_bulk_training(
    input_path: Path,
    output_dir: Path,
    execute_paper: bool = False,
    max_orders: int = 25,
) -> List[Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    signals = _split_signals(input_path.read_text(encoding="utf-8"))
    prediction_cache: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    alpaca = AlpacaPaperClient()
    orders_attempted = 0

    for index, signal in enumerate(signals, start=1):
        routed = classify_and_parse(signal)
        row: Dict[str, Any] = {
            "index": index,
            "input": signal.replace("\n", " / "),
            "kind": routed.kind,
            "parse_valid": routed.kind in ("EQUITY", "OPTION"),
            "parse_reason": routed.reason,
            "user_action": "",
            "symbol": "",
            "quantity": "",
            "option_strike": "",
            "option_side": "",
            "option_expiry": "",
            "option_structure": "",
            "option_tense": "",
            "prediction_status": "",
            "ai_decision": "",
            "predicted_return_pct": "",
            "confidence": "",
            "risk": "",
            "score": "",
            "market_regime": "",
            "final_output": "",
            "agent_reason": routed.reason,
            "paper_execution": "DRY_RUN",
            "paper_order_id": "",
            "paper_execution_detail": "",
        }

        if routed.kind == "INVALID":
            row["final_output"] = "INVALID"
            rows.append(row)
            print(f"[{index}/{len(signals)}] INVALID - {routed.reason}", flush=True)
            continue

        if routed.kind == "NO_TRADE":
            row["final_output"] = "NO_TRADE"
            rows.append(row)
            print(f"[{index}/{len(signals)}] NO_TRADE - {routed.reason}", flush=True)
            continue

        if routed.kind == "OPTION":
            option = routed.option
            row.update({
                "user_action": option.tense,
                "symbol": option.root,
                "quantity": _resolved_option_qty(option) if option.valid else "",
                "option_strike": option.strike if option.strike is not None else "",
                "option_side": option.side or "",
                "option_expiry": option.expiry_date or "",
                "option_structure": option.structure or "",
                "option_tense": option.tense,
            })

            if not option.valid:
                row["final_output"] = "OPTION_INCOMPLETE"
                rows.append(row)
                print(f"[{index}/{len(signals)}] OPTION_INCOMPLETE - {option.reason}", flush=True)
                continue

            if option.is_multi_leg and not (option.strike and option.side):
                row["final_output"] = "OPTION_MULTI_LEG_TRACKED"
                rows.append(row)
                print(f"[{index}/{len(signals)}] {option.root} multi-leg {option.structure} tracked only", flush=True)
                continue

            if option.tense in ("past", "unknown"):
                row["final_output"] = "OPTION_JOURNAL"
                rows.append(row)
                print(f"[{index}/{len(signals)}] {option.root} journaled (tense={option.tense})", flush=True)
                continue

            underlying = resolve_underlying_for_prediction(option.root)
            if underlying not in prediction_cache:
                print(f"[{index}/{len(signals)}] Fetching prediction for {underlying} (option underlying)...", flush=True)
                prediction_cache[underlying] = run_project_prediction(underlying)
            else:
                print(f"[{index}/{len(signals)}] Using cached prediction for {underlying}.", flush=True)
            prediction = prediction_cache[underlying]
            row["prediction_status"] = prediction.get("status", "")

            if prediction.get("status") != "SUCCESS":
                row["final_output"] = "HOLD"
                row["agent_reason"] = f"Prediction failed: {prediction.get('error', 'Unknown error')}"
                rows.append(row)
                print(f"[{index}/{len(signals)}] {underlying} prediction failed -> HOLD", flush=True)
                continue

            ai_prediction = prediction.get("ai_prediction") or {}
            mapped_action = "BUY" if option.side == "CALL" else "SELL"
            decision = _evaluate_final_action(mapped_action, ai_prediction)
            row.update({
                "ai_decision": str(ai_prediction.get("decision") or "REVIEW").upper(),
                "predicted_return_pct": round(decision.predicted_return, 4),
                "confidence": round(decision.confidence, 4),
                "risk": round(decision.risk, 4),
                "score": round(decision.score, 4),
                "market_regime": decision.market_regime,
                "final_output": f"OPTION_{decision.action}" if decision.action == mapped_action else "HOLD",
                "agent_reason": decision.reason,
            })
            add_decision_history({
                "source": "bulk_training", "symbol": option.root, "kind": "OPTION",
                "user_action": mapped_action, "ai_decision": row["ai_decision"],
                "final_action": decision.action,
                "predicted_return_pct": round(decision.predicted_return, 4),
                "confidence": round(decision.confidence, 4), "risk": round(decision.risk, 4),
                "score": round(decision.score, 4), "market_regime": decision.market_regime,
                "reason": decision.reason, "raw_input": signal,
            })
            # Live option order placement stays in discord_agent.py's single-signal
            # path (has_options_trading -> get_option_contracts -> submit_option_order).
            # The bulk tool only classifies + AI-gates options in this first version.
            if execute_paper and row["final_output"].startswith("OPTION_"):
                row["paper_execution"] = "NOT_IMPLEMENTED_IN_BULK_TOOL"
                row["paper_execution_detail"] = "Use the live Discord bot to place approved option orders."
            print(
                f"[{index}/{len(signals)}] {option.root} {option.side} -> {row['final_output']} "
                f"(score {decision.score:.1f})",
                flush=True,
            )
            rows.append(row)
            continue

        # EQUITY
        parsed = routed.equity
        row.update({
            "user_action": parsed.action,
            "symbol": parsed.symbol,
            "quantity": parsed.quantity if parsed.quantity is not None else "",
        })

        if parsed.symbol not in prediction_cache:
            print(f"[{index}/{len(signals)}] Fetching prediction for {parsed.symbol}...", flush=True)
            prediction_cache[parsed.symbol] = run_project_prediction(parsed.symbol)
        else:
            print(f"[{index}/{len(signals)}] Using cached prediction for {parsed.symbol}.", flush=True)
        prediction = prediction_cache[parsed.symbol]
        row["prediction_status"] = prediction.get("status", "")

        if prediction.get("status") != "SUCCESS":
            row["final_output"] = "HOLD"
            row["agent_reason"] = f"Prediction failed: {prediction.get('error', 'Unknown error')}"
            rows.append(row)
            print(f"[{index}/{len(signals)}] {parsed.symbol} prediction failed -> HOLD", flush=True)
            continue

        ai_prediction = prediction.get("ai_prediction") or {}
        decision = _evaluate_final_action(parsed.action, ai_prediction)
        row.update(
            {
                "ai_decision": str(ai_prediction.get("decision") or "REVIEW").upper(),
                "predicted_return_pct": round(decision.predicted_return, 4),
                "confidence": round(decision.confidence, 4),
                "risk": round(decision.risk, 4),
                "score": round(decision.score, 4),
                "market_regime": decision.market_regime,
                "final_output": decision.action,
                "agent_reason": decision.reason,
            }
        )
        add_decision_history(
            {
                "source": "bulk_training",
                "symbol": parsed.symbol,
                "user_action": parsed.action,
                "ai_decision": row["ai_decision"],
                "final_action": decision.action,
                "predicted_return_pct": round(decision.predicted_return, 4),
                "confidence": round(decision.confidence, 4),
                "risk": round(decision.risk, 4),
                "score": round(decision.score, 4),
                "market_regime": decision.market_regime,
                "reason": decision.reason,
                "raw_input": signal,
            }
        )
        if execute_paper and decision.action in {"BUY", "SELL"}:
            if orders_attempted >= max_orders:
                row["paper_execution"] = "SKIPPED"
                row["paper_execution_detail"] = f"Max paper order limit reached ({max_orders})."
            else:
                execution = _execute_paper_trade(alpaca, decision.action, parsed.symbol, parsed.quantity)
                orders_attempted += 1
                row["paper_execution"] = execution["status"]
                row["paper_order_id"] = execution["order_id"]
                row["paper_execution_detail"] = execution["detail"]
        print(
            f"[{index}/{len(signals)}] {parsed.symbol} {parsed.action} -> {decision.action} "
            f"(score {decision.score:.1f})",
            flush=True,
        )
        rows.append(row)

    csv_path = output_dir / "bulk_signal_training_results.csv"
    md_path = output_dir / "bulk_signal_training_results.md"
    fields = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Bulk Signal Training Results\n\n")
        f.write("| # | Input | Kind | Symbol | Option (side/strike/expiry) | User | AI Return | Conf/Risk/Score | Output | Paper | Reason |\n")
        f.write("|---:|---|---|---|---|---|---:|---|---|---|---|\n")
        for row in rows:
            reason = str(row["agent_reason"]).replace("|", "/")[:220]
            inp = str(row["input"]).replace("|", "/")[:120]
            paper = str(row["paper_execution"]).replace("|", "/")
            option_bits = (
                f"{row['option_side']}/{row['option_strike']}/{row['option_expiry']}"
                if row["kind"] == "OPTION" else ""
            )
            f.write(
                f"| {row['index']} | {inp} | {row['kind']} | {row['symbol']} | {option_bits} | "
                f"{row['user_action']} | {row['predicted_return_pct']} | "
                f"{row['confidence']}/{row['risk']}/{row['score']} | {row['final_output']} | {paper} | {reason} |\n"
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("discord_stock_prediction_agent") / "training_reports")
    parser.add_argument("--execute-paper", action="store_true", help="Place Alpaca paper trades for applicable final BUY/SELL outputs.")
    parser.add_argument("--max-orders", type=int, default=25, help="Maximum paper orders to attempt in one bulk run.")
    args = parser.parse_args()
    rows = run_bulk_training(args.input_file, args.output_dir, execute_paper=args.execute_paper, max_orders=args.max_orders)
    counts: Dict[str, int] = {}
    execution_counts: Dict[str, int] = {}
    for row in rows:
        counts[str(row["final_output"])] = counts.get(str(row["final_output"]), 0) + 1
        execution_counts[str(row["paper_execution"])] = execution_counts.get(str(row["paper_execution"]), 0) + 1
    print(f"Processed {len(rows)} signal(s). Output counts: {counts}")
    print(f"Paper execution counts: {execution_counts}")
    print(f"Reports written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
