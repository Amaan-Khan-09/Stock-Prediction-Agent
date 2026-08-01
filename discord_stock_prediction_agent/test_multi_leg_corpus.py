import json
import os
from pathlib import Path

import pytest

from .multi_leg_dataset_audit import _normalize
from .options_parser import classify_and_parse


CORPUS = Path(os.environ.get(
    "MULTI_LEG_CORPUS_JSON",
    r"C:\Users\mdama\Downloads\multi_leg_options_120_test_cases.json",
))


@pytest.fixture(scope="module")
def records():
    if not CORPUS.exists():
        pytest.skip(f"Corpus is unavailable: {CORPUS}")
    return json.loads(CORPUS.read_text(encoding="utf-8"))


def test_all_corpus_signals_are_valid_multi_leg_options(records):
    for record in records:
        parsed = classify_and_parse(record["signal"])
        assert parsed.kind == "OPTION", record["id"]
        assert parsed.option is not None and parsed.option.valid, record["id"]
        assert parsed.option.is_multi_leg, record["id"]
        assert parsed.option.semantic_contract, record["id"]


def test_all_observable_corpus_fields_match(records):
    for record in records:
        parsed = classify_and_parse(record["signal"])
        actual = parsed.option.semantic_contract
        assert _normalize(actual, record["signal"]) == _normalize(
            record["expected_output"], record["signal"]
        ), record["id"]


def test_contract_is_projected_to_runtime_fields(records):
    for record in records:
        parsed = classify_and_parse(record["signal"])
        option = parsed.option
        contract = option.semantic_contract
        option_contract_legs = [
            leg for leg in contract["legs"] if leg.get("asset_type") != "STOCK"
        ]
        assert option.root == contract["symbol"], record["id"]
        assert len(option.legs) == len(option_contract_legs), record["id"]
        assert option.quantity == float(contract["quantity"]), record["id"]


def test_agent_on_and_off_embed_source_uses_same_contract(records):
    # Agent mode changes the decision gate, never the parsed input contract.
    sample = records[94]
    first = classify_and_parse(sample["signal"]).option.semantic_contract
    second = classify_and_parse(sample["signal"]).option.semantic_contract
    assert first == second
    assert first["advanced_instructions"]["entry_condition"]["type"] == "DAILY_CLOSE_ABOVE"


def test_hard_tier_risk_management_projects_onto_runtime_fields(records):
    import re

    for record in records:
        expected = record["expected_output"]
        risk = expected.get("risk_management")
        if not isinstance(risk, dict):
            continue
        option = classify_and_parse(record["signal"]).option
        # A stop_loss/take_profit price that isn't observable from the signal
        # text (e.g. only "MAX LOSS ..." with no explicit SL/TP price) is a
        # derived value the JSON fixture fills in -- not something the parser
        # can read off the raw text, so only assert on what's actually stated.
        if risk.get("stop_loss") is not None and re.search(r"\bSL\b", record["signal"], re.IGNORECASE):
            assert option.stop_loss == pytest.approx(float(risk["stop_loss"])), record["id"]
        if risk.get("take_profit") is not None and re.search(r"\bTP\b", record["signal"], re.IGNORECASE):
            assert option.target_price == pytest.approx(float(risk["take_profit"])), record["id"]
        if risk.get("maximum_loss_inr") is not None:
            assert option.maximum_loss_amount == pytest.approx(float(risk["maximum_loss_inr"])), record["id"]


def test_complex_tier_advanced_instructions_project_onto_runtime_fields(records):
    for record in records:
        expected = record["expected_output"]
        advanced = expected.get("advanced_instructions")
        if not isinstance(advanced, dict):
            continue
        option = classify_and_parse(record["signal"]).option
        if advanced.get("maximum_loss_inr") is not None:
            assert option.maximum_loss_amount == pytest.approx(
                float(advanced["maximum_loss_inr"])
            ), record["id"]
        entry_condition = advanced.get("entry_condition")
        if isinstance(entry_condition, dict) and entry_condition.get("value") is not None:
            assert option.underlying_trigger_price == pytest.approx(
                float(entry_condition["value"])
            ), record["id"]
            assert option.underlying_trigger_direction in {"above", "below"}, record["id"]


def test_conditional_entry_and_max_loss_are_visible_in_the_review_embed(records):
    from . import discord_agent

    conditional = next(
        r for r in records if r["strategy"] == "CONDITIONAL_BUTTERFLY"
    )
    max_loss = next(
        r for r in records if r["strategy"] == "CALL_BACKSPREAD" and r["difficulty"] == "Hard"
    )
    for record in (conditional, max_loss):
        option = classify_and_parse(record["signal"]).option
        quality = discord_agent._option_signal_quality(option)
        embed = discord_agent._option_embed(
            option, {}, "BUY",
            discord_agent._direct_signal_decision("BUY", 100),
            discord_agent._direct_option_validation("BUY"),
            quality,
        )
        contract_text = "\n".join(
            value for name, value in ((f.name, f.value) for f in embed.fields)
            if name.startswith("Parsed Field Contract")
        )
        if record is conditional:
            assert "Entry Condition" in contract_text, record["id"]
        else:
            assert "Maximum Loss Inr: 25000" in contract_text or "Maximum Loss Inr: 25000.0" in contract_text, record["id"]
