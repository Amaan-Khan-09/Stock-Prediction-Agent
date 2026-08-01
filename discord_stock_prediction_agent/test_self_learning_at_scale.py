"""Proof that the agent's pattern-based self-learning overlay actually
changes future decisions, plus a 1,000+-signal bulk training pass that
exercises the exact learning pipeline discord_agent.py uses (parse ->
features -> record -> recall) with no real network/AI/broker calls.

_apply_learning_overlay is the function discord_agent.py calls after every
BUY/SELL decision to fold in accumulated pattern history. It had never been
exercised end-to-end anywhere in the test suite (test_learning_store.py only
checks the raw pattern counters; test_decision_contract_learning.py only
checks feature extraction) -- this file closes that gap.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from . import discord_agent, state_store
from .options_parser import classify_and_parse
from .test_real_world_signal_corpus import build_corpus


def _isolated_state():
    tmp = tempfile.TemporaryDirectory()
    original = state_store.STATE_PATH
    state_store.STATE_PATH = Path(tmp.name) / "agent_state.json"
    return tmp, original


def _restore_state(tmp, original) -> None:
    state_store.STATE_PATH = original
    tmp.cleanup()


def test_self_learning_overlay_shifts_future_decisions_after_training() -> None:
    tmp, original = _isolated_state()
    try:
        good_features = discord_agent._learning_features("BUY AAPL QTY 5", "equity", "BUY", "AAPL")
        bad_features = discord_agent._learning_features("SELL MSFT QTY 5", "equity", "SELL", "MSFT")

        for i in range(20):
            approved = i < 18  # 90% approval -> should earn a confidence boost
            state_store.record_learning_event(
                good_features,
                {
                    "final_action": "BUY" if approved else "HOLD",
                    "score": 80 if approved else 30,
                    "predicted_return_pct": 1.0,
                },
            )
        for i in range(20):
            approved = i < 4  # 20% approval -> should trigger learned caution
            state_store.record_learning_event(
                bad_features,
                {
                    "final_action": "SELL" if approved else "HOLD",
                    "score": 75 if approved else 20,
                    "predicted_return_pct": -0.5,
                },
            )

        baseline = discord_agent.DecisionResult("BUY", "AI approved", 1.2, 80, 20, 70, "trend")
        boosted = discord_agent._apply_learning_overlay(baseline, good_features)
        assert boosted.action == "BUY", boosted
        assert boosted.score > baseline.score, (boosted.score, baseline.score)
        assert boosted.score <= 100.0

        risky = discord_agent.DecisionResult("SELL", "AI approved", -1.0, 75, 60, 70, "trend")
        cautioned = discord_agent._apply_learning_overlay(risky, bad_features)
        assert cautioned.action == "HOLD", cautioned
        assert cautioned.score < risky.score, (cautioned.score, risky.score)
    finally:
        _restore_state(tmp, original)


def test_below_min_samples_learning_overlay_is_a_no_op() -> None:
    tmp, original = _isolated_state()
    try:
        features = discord_agent._learning_features("BUY NVDA QTY 3", "equity", "BUY", "NVDA")
        for _ in range(3):  # below config.learning_min_samples (default 5)
            state_store.record_learning_event(
                features, {"final_action": "BUY", "score": 90, "predicted_return_pct": 2.0}
            )
        decision = discord_agent.DecisionResult("BUY", "AI approved", 1.0, 70, 20, 60, "trend")
        result = discord_agent._apply_learning_overlay(decision, features)
        assert result == decision, "overlay must not act before learning_min_samples is reached"
    finally:
        _restore_state(tmp, original)


def _features_for(expected, routed):
    if routed.kind == "EQUITY" and routed.equity and routed.equity.valid:
        return discord_agent._learning_features(
            expected.text, "equity", routed.equity.action, routed.equity.symbol, equity=routed.equity
        )
    if routed.kind == "OPTION" and routed.option and routed.option.valid:
        option = routed.option
        mapped_action = (
            discord_agent._multi_leg_mapped_action(option)
            if option.is_multi_leg
            else discord_agent._option_mapped_action(option)
        )
        return discord_agent._learning_features(
            expected.text,
            "option_mleg" if option.is_multi_leg else "option",
            mapped_action,
            option.root,
            option,
        )
    return None


def test_bulk_training_pass_over_1000_plus_real_world_signals() -> None:
    tmp, original = _isolated_state()
    try:
        corpus = build_corpus()
        assert len(corpus) >= 1_000, f"corpus too small to train at scale: {len(corpus)}"

        trained = 0
        for index, expected in enumerate(corpus):
            routed = classify_and_parse(expected.text)
            features = _features_for(expected, routed)
            if features is None:
                continue

            # Deterministic, varied synthetic outcome so every branch of
            # record_learning_event (approved/blocked/reviewed) gets
            # exercised at scale -- no prediction API or broker call involved.
            bucket = index % 5
            final_action = features["action"] if bucket < 3 else ("HOLD" if bucket == 3 else "REVIEW")
            state_store.record_learning_event(
                features,
                {
                    "final_action": final_action,
                    "score": 85.0 if bucket < 3 else 35.0,
                    "predicted_return_pct": 1.5 if bucket < 3 else -0.8,
                },
            )
            trained += 1

        assert trained >= 1_000, f"expected to train on at least 1000 tradeable signals, got {trained}"

        state = state_store.load_state()
        patterns = state.get("signal_learning", {})
        assert patterns, "training must have produced at least one learned pattern"
        total_seen = sum(int(p.get("seen") or 0) for p in patterns.values())
        assert total_seen == trained, (total_seen, trained)
        for pattern in patterns.values():
            assert (
                int(pattern["approved"]) + int(pattern["blocked"]) + int(pattern["reviewed"])
                == int(pattern["seen"])
            ), pattern

        summary = state_store.get_signal_learning_summary(limit=len(patterns))
        assert summary["total_patterns"] == len(patterns)
        print(f"TRAINED ON {trained} SIGNALS ACROSS {summary['total_patterns']} LEARNED PATTERNS")
    finally:
        _restore_state(tmp, original)


def run_all() -> None:
    test_self_learning_overlay_shifts_future_decisions_after_training()
    test_below_min_samples_learning_overlay_is_a_no_op()
    test_bulk_training_pass_over_1000_plus_real_world_signals()
    print("SELF-LEARNING AT SCALE TESTS PASSED")


if __name__ == "__main__":
    run_all()
