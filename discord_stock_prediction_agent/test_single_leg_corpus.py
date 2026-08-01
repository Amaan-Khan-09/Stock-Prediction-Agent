import json
import os
from pathlib import Path

import pytest

from .options_parser import classify_and_parse


CORPUS = Path(os.environ.get(
    "SINGLE_LEG_CORPUS_JSON",
    r"C:\Users\mdama\Downloads\single_leg_options_120_test_cases.json",
))


@pytest.fixture(scope="module")
def records():
    if not CORPUS.exists():
        pytest.skip(f"Corpus is unavailable: {CORPUS}")
    return json.loads(CORPUS.read_text(encoding="utf-8"))


def test_all_corpus_signals_are_valid_single_leg_options(records):
    for record in records:
        parsed = classify_and_parse(record["signal"])
        assert parsed.kind == "OPTION", record["id"]
        assert parsed.option is not None and parsed.option.valid, record["id"]
        assert not parsed.option.is_multi_leg, record["id"]
        assert parsed.option.semantic_contract, record["id"]


def test_all_corpus_fields_match_expected_output_exactly(records):
    for record in records:
        parsed = classify_and_parse(record["signal"])
        actual = parsed.option.semantic_contract
        assert actual == record["expected_output"], record["id"]


def test_contract_is_projected_to_runtime_fields(records):
    for record in records:
        option = classify_and_parse(record["signal"]).option
        contract = option.semantic_contract
        assert option.root == contract["symbol"], record["id"]
        if contract.get("stop_loss") is not None:
            assert option.stop_loss == pytest.approx(float(contract["stop_loss"])), record["id"]
        if contract.get("maximum_loss") is not None:
            assert option.maximum_loss_amount == pytest.approx(float(contract["maximum_loss"])), record["id"]
        if contract.get("time_in_force"):
            assert option.time_in_force == str(contract["time_in_force"]), record["id"]


def test_agent_on_and_off_embed_source_uses_same_contract(records):
    # Agent mode changes the decision gate, never the parsed input contract.
    for record in records:
        first = classify_and_parse(record["signal"]).option.semantic_contract
        second = classify_and_parse(record["signal"]).option.semantic_contract
        assert first == second, record["id"]
