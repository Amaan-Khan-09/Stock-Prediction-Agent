"""Audit the 120-case multi-leg corpus without submitting broker orders."""
from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from .options_parser import classify_and_parse


STRATEGY_ALIASES = {
    "CALL_DEBIT_SPREAD": "BULL_CALL_SPREAD",
    "PUT_DEBIT_SPREAD": "BEAR_PUT_SPREAD",
}


def _quantity_is_observable(signal: str) -> bool:
    return bool(re.search(r"\bQTY\s*\d+|\bOPEN\s+\d+\s+[A-Z]", signal, re.IGNORECASE))


def _normalize(contract: dict[str, Any], signal: str) -> dict[str, Any]:
    value = deepcopy(contract)
    value["strategy"] = STRATEGY_ALIASES.get(value.get("strategy"), value.get("strategy"))
    if not _quantity_is_observable(signal):
        value.pop("quantity", None)
    risk = value.get("risk_management")
    if isinstance(risk, dict) and not re.search(r"\bSL\b", signal, re.IGNORECASE):
        risk["stop_loss"] = None
    return value


def audit(json_path: Path) -> dict[str, Any]:
    records = json.loads(json_path.read_text(encoding="utf-8"))
    exact = 0
    observable = 0
    failures: list[dict[str, Any]] = []
    for record in records:
        parsed = classify_and_parse(record["signal"])
        actual = parsed.option.semantic_contract if parsed.option else {}
        expected = record["expected_output"]
        if actual == expected:
            exact += 1
        if _normalize(actual, record["signal"]) == _normalize(expected, record["signal"]):
            observable += 1
        else:
            failures.append({
                "id": record["id"], "strategy": record["strategy"],
                "signal": record["signal"], "actual": actual, "expected": expected,
            })
    return {
        "total": len(records), "exact_matches": exact,
        "observable_matches": observable, "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = audit(args.json_path)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "failures"}, indent=2))
    if report["failures"]:
        print("Observable mismatches:", [item["id"] for item in report["failures"]])
    return 0 if report["observable_matches"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
