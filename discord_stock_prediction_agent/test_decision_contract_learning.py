"""Decision-contract and learning regressions across every supported asset path."""
from __future__ import annotations

from . import discord_agent, state_store
from .options_parser import classify_and_parse
from .signal_parser import parse_signal


def _embed_fields(embed) -> dict[str, str]:
    return {field.name: field.value for field in embed.fields}


def test_equity_agent_modes_render_the_same_normalized_contract() -> None:
    parsed = parse_signal("SELL AMZN Qty 20 STOP 245.00 LIMIT 244.50 GTC")
    assert parsed.valid
    assert parsed.order_intent

    direct = discord_agent._direct_equity_embed(
        parsed, discord_agent._direct_signal_decision("SELL")
    )
    prediction = {
        "status": "SUCCESS",
        "ai_prediction": {
            "decision": "SELL",
            "predicted_return_pct": -1.0,
            "confidence_score": 80,
            "risk_score": 70,
        },
    }
    agent_on = discord_agent._prediction_embed(
        parsed,
        prediction,
        discord_agent.DecisionResult("SELL", "test", -1, 80, 70, 75, "test"),
    )

    direct_contract = _embed_fields(direct)["Parsed Signal"]
    agent_contract = _embed_fields(agent_on)["Parsed Signal"]
    assert direct_contract == agent_contract
    assert "Action: SELL | Symbol: AMZN" in direct_contract
    assert "Quantity: 20.0" in direct_contract
    assert "Order Type: STOP_LIMIT" in direct_contract
    assert "Time In Force: GTC" in direct_contract
    assert "Limit Price: $244.50 | Stop Price: $245.00" in direct_contract


def test_learning_features_include_the_complete_equity_contract() -> None:
    parsed = parse_signal("BUY GOOGL Qty 50 LIMIT 218.75 GTC")
    features = discord_agent._learning_features(
        parsed.raw_text, "equity", parsed.action, parsed.symbol, equity=parsed
    )
    assert features["strategy"] == "stock_order"
    assert features["order_type"] == "LIMIT"
    assert features["contract_status"] == "VALID"
    assert {"limit_price", "quantity", "time_in_force"}.issubset(
        features["semantic_fields"]
    )


def test_learning_identity_separates_single_and_multi_leg_strategies() -> None:
    single = classify_and_parse("BTO AAPL 240C 09/19 @3.45 Qty 2 LIMIT").option
    multi = classify_and_parse(
        "BTO AAPL 240C / STO AAPL 250C 09/19 @4.60 Debit Qty 2"
    ).option
    assert single and multi

    single_features = discord_agent._learning_features(
        single.raw_text, "option", "BUY", single.root, single
    )
    multi_features = discord_agent._learning_features(
        multi.raw_text, "option_mleg", "BUY", multi.root, multi
    )
    assert single_features["leg_count"] == 1
    assert multi_features["leg_count"] == 2
    assert multi_features["strategy"]
    assert state_store._learning_key(single_features) != state_store._learning_key(
        multi_features
    )
