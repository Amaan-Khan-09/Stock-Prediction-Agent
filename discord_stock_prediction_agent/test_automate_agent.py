"""Unit tests for automate_agent.py's pure decision logic.

No Alpaca, no Discord, no Gemini -- these test the actual decisions
(which candidate to buy, which existing position to evict) as plain
synchronous functions, matching the rest of this codebase's pattern of
keeping decision logic separate from I/O.

Run:
    python -m discord_stock_prediction_agent.test_automate_agent
"""
from __future__ import annotations

from .automate_agent import (
    AUTOMATE_AGENT_TAG,
    BoomCandidate,
    confidence_scaled_risk_multiplier,
    count_automate_positions,
    oldest_automate_position,
    plan_automate_trades,
    rank_boom_candidates,
)

PASS = 0
FAIL = 0


def _assert(condition: bool, name: str, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}" + (f" -- {detail}" if detail else ""))


def _position(symbol: str, opened_by: str = AUTOMATE_AGENT_TAG, updated_at: str = "2026-01-01T00:00:00Z") -> dict:
    return {"symbol": symbol, "opened_by": opened_by, "updated_at": updated_at, "qty": 1}


def test_rank_keeps_only_buy_decisions() -> None:
    print("\nTest: ranking keeps only BUY-decision candidates")
    candidates = [
        BoomCandidate("AAPL", "BUY"),
        BoomCandidate("MSFT", "HOLD"),
        BoomCandidate("NVDA", "SELL"),
        BoomCandidate("TSLA", "REVIEW"),
    ]
    ranked = rank_boom_candidates(candidates)
    _assert([c.symbol for c in ranked] == ["AAPL"], "only AAPL (the BUY) survives", str(ranked))


def test_rank_orders_by_confidence_descending() -> None:
    print("\nTest: ranking sorts BUY candidates by confidence, strongest first")
    candidates = [
        BoomCandidate("AAPL", "BUY", confidence=0.4),
        BoomCandidate("MSFT", "BUY", confidence=0.9),
        BoomCandidate("NVDA", "BUY", confidence=0.6),
    ]
    ranked = rank_boom_candidates(candidates)
    _assert([c.symbol for c in ranked] == ["MSFT", "NVDA", "AAPL"], "sorted strongest-first", str(ranked))


def test_rank_tiebreaks_by_predicted_return_pct() -> None:
    print("\nTest: ranking tiebreaks equal-confidence candidates by predicted_return_pct")
    candidates = [
        BoomCandidate("AAPL", "BUY", confidence=0.7, predicted_return_pct=1.0),
        BoomCandidate("MSFT", "BUY", confidence=0.7, predicted_return_pct=3.5),
        BoomCandidate("NVDA", "BUY", confidence=0.7, predicted_return_pct=2.0),
    ]
    ranked = rank_boom_candidates(candidates)
    _assert(
        [c.symbol for c in ranked] == ["MSFT", "NVDA", "AAPL"],
        "same confidence, higher predicted_return_pct wins",
        str(ranked),
    )


def test_rank_excludes_needs_human_review() -> None:
    print("\nTest: ranking drops BUY candidates the model itself flagged as uncertain")
    candidates = [
        BoomCandidate("AAPL", "BUY", confidence=0.9, needs_human_review=True),
        BoomCandidate("MSFT", "BUY", confidence=0.5, needs_human_review=False),
    ]
    ranked = rank_boom_candidates(candidates)
    _assert(
        [c.symbol for c in ranked] == ["MSFT"],
        "AAPL excluded despite higher confidence -- needs_human_review is a hard filter, not a ranking input",
        str(ranked),
    )


def test_rank_excludes_below_min_confidence() -> None:
    print("\nTest: ranking drops BUY candidates below an explicit min_confidence floor")
    candidates = [
        BoomCandidate("AAPL", "BUY", confidence=90),
        BoomCandidate("MSFT", "BUY", confidence=59),
        BoomCandidate("NVDA", "BUY", confidence=60),
    ]
    ranked = rank_boom_candidates(candidates, min_confidence=60)
    _assert(
        [c.symbol for c in ranked] == ["AAPL", "NVDA"],
        "MSFT (below 60) excluded, NVDA (exactly at 60) kept",
        str(ranked),
    )


def test_plan_respects_min_confidence() -> None:
    print("\nTest: plan_automate_trades threads min_confidence through to ranking")
    candidates = [BoomCandidate("AAPL", "BUY", confidence=90), BoomCandidate("MSFT", "BUY", confidence=10)]
    plan = plan_automate_trades(
        candidates, open_positions=[], min_positions=1, max_positions=5, min_confidence=60
    )
    _assert(plan.to_buy == ["AAPL"], "only the >=60-confidence candidate survives", str(plan.to_buy))


def test_confidence_scaled_risk_multiplier_floor_at_low_confidence() -> None:
    print("\nTest: confidence-scaled sizing floors out at/below confidence 50")
    _assert(confidence_scaled_risk_multiplier(50) == 0.75, "exactly at the floor threshold")
    _assert(confidence_scaled_risk_multiplier(10) == 0.75, "well below floor threshold, still clamped")
    _assert(confidence_scaled_risk_multiplier(-5) == 0.75, "negative/malformed input never goes below floor")


def test_confidence_scaled_risk_multiplier_caps_at_high_confidence() -> None:
    print("\nTest: confidence-scaled sizing caps out at/above confidence 100")
    _assert(confidence_scaled_risk_multiplier(100) == 1.25, "exactly at the cap threshold")
    _assert(confidence_scaled_risk_multiplier(150) == 1.25, "above 100 still clamped to the cap")


def test_confidence_scaled_risk_multiplier_interpolates_linearly() -> None:
    print("\nTest: confidence-scaled sizing interpolates between floor and cap")
    mid = confidence_scaled_risk_multiplier(75)
    _assert(abs(mid - 1.0) < 1e-9, "confidence 75 (midpoint) is exactly the neutral 1.0x", str(mid))
    high = confidence_scaled_risk_multiplier(90)
    _assert(0.75 < mid < high < 1.25, "monotonically increasing between floor and cap", f"mid={mid} high={high}")


def test_oldest_automate_position_ignores_user_trades() -> None:
    print("\nTest: eviction never targets a real user's position")
    positions = [
        _position("AAPL", opened_by="", updated_at="2026-01-01T00:00:00Z"),  # user trade, oldest
        _position("MSFT", opened_by=AUTOMATE_AGENT_TAG, updated_at="2026-01-02T00:00:00Z"),
        _position("NVDA", opened_by=AUTOMATE_AGENT_TAG, updated_at="2026-01-01T12:00:00Z"),
    ]
    evict = oldest_automate_position(positions)
    _assert(evict == "NVDA", "picks the oldest automate_agent position, not the user's older AAPL", str(evict))


def test_oldest_automate_position_none_when_all_user_owned() -> None:
    print("\nTest: no eviction candidate when every open position is a real user's")
    positions = [_position("AAPL", opened_by=""), _position("MSFT", opened_by="")]
    _assert(oldest_automate_position(positions) is None, "returns None")


def test_count_automate_positions() -> None:
    print("\nTest: count only counts automate_agent-tagged positions")
    positions = [
        _position("AAPL", opened_by=AUTOMATE_AGENT_TAG),
        _position("MSFT", opened_by=""),
        _position("NVDA", opened_by=AUTOMATE_AGENT_TAG),
    ]
    _assert(count_automate_positions(positions) == 2, "counts 2", str(count_automate_positions(positions)))


def test_plan_buys_up_to_available_slots_no_eviction_needed() -> None:
    print("\nTest: plan buys into open slots without evicting anything")
    candidates = [BoomCandidate("AAPL", "BUY"), BoomCandidate("MSFT", "BUY")]
    plan = plan_automate_trades(candidates, open_positions=[], min_positions=1, max_positions=5)
    _assert(plan.to_evict == [], "nothing evicted")
    _assert(set(plan.to_buy) == {"AAPL", "MSFT"}, "buys both candidates", str(plan.to_buy))


def test_plan_never_exceeds_max_positions() -> None:
    print("\nTest: plan never exceeds the max position cap even with many candidates")
    candidates = [BoomCandidate(f"SYM{i}", "BUY") for i in range(10)]
    plan = plan_automate_trades(candidates, open_positions=[], min_positions=1, max_positions=5)
    _assert(len(plan.to_buy) == 5, "buys exactly 5, not 10", str(len(plan.to_buy)))


def test_plan_evicts_oldest_when_at_cap() -> None:
    print("\nTest: at the cap, plan evicts the oldest automate_agent position to make room")
    existing = [
        _position("OLD1", updated_at="2026-01-01T00:00:00Z"),
        _position("OLD2", updated_at="2026-01-02T00:00:00Z"),
        _position("OLD3", updated_at="2026-01-03T00:00:00Z"),
        _position("OLD4", updated_at="2026-01-04T00:00:00Z"),
        _position("OLD5", updated_at="2026-01-05T00:00:00Z"),
    ]
    candidates = [BoomCandidate("NEW1", "BUY")]
    plan = plan_automate_trades(candidates, existing, min_positions=1, max_positions=5)
    _assert(plan.to_evict == ["OLD1"], "evicts the oldest (OLD1)", str(plan.to_evict))
    _assert(plan.to_buy == ["NEW1"], "buys the new candidate", str(plan.to_buy))


def test_plan_caps_evictions_per_cycle_even_with_many_fresh_candidates() -> None:
    print("\nTest: at the cap, plan evicts at most max_evictions_per_cycle positions in one call")
    existing = [
        _position("OLD1", updated_at="2026-01-01T00:00:00Z"),
        _position("OLD2", updated_at="2026-01-02T00:00:00Z"),
        _position("OLD3", updated_at="2026-01-03T00:00:00Z"),
        _position("OLD4", updated_at="2026-01-04T00:00:00Z"),
        _position("OLD5", updated_at="2026-01-05T00:00:00Z"),
    ]
    # 5 fresh, higher-ranked BUY candidates -- enough to justify swapping the
    # entire book if nothing capped it.
    candidates = [BoomCandidate(f"NEW{i}", "BUY", confidence=90) for i in range(1, 6)]
    plan = plan_automate_trades(candidates, existing, min_positions=1, max_positions=5)
    _assert(
        len(plan.to_evict) == 1,
        "only 1 eviction per cycle by default, not the entire 5-position book",
        str(plan.to_evict),
    )
    _assert(plan.to_evict == ["OLD1"], "still evicts the oldest first", str(plan.to_evict))
    _assert(len(plan.to_buy) == 1, "only buys as many as slots freed up", str(plan.to_buy))

    # An explicit higher cap allows more turnover in one cycle if configured.
    plan_uncapped = plan_automate_trades(
        candidates, existing, min_positions=1, max_positions=5, max_evictions_per_cycle=5
    )
    _assert(
        len(plan_uncapped.to_evict) == 5,
        "a raised max_evictions_per_cycle allows more evictions in one call",
        str(plan_uncapped.to_evict),
    )


def test_plan_never_evicts_a_real_user_position_even_at_cap() -> None:
    print("\nTest: at the cap, plan never evicts a real user's position to make room")
    existing = [
        _position("USERPICK", opened_by="", updated_at="2020-01-01T00:00:00Z"),  # oldest overall, but user-owned
        _position("A1", updated_at="2026-01-01T00:00:00Z"),
        _position("A2", updated_at="2026-01-02T00:00:00Z"),
        _position("A3", updated_at="2026-01-03T00:00:00Z"),
        _position("A4", updated_at="2026-01-04T00:00:00Z"),
    ]
    # Only 4 automate_agent positions among 5 total -- one slot is free already.
    candidates = [BoomCandidate("NEW1", "BUY")]
    plan = plan_automate_trades(candidates, existing, min_positions=1, max_positions=5)
    _assert(plan.to_evict == [], "no eviction needed, a slot was already free")
    _assert("USERPICK" not in plan.to_evict, "USERPICK is never touched")


def test_plan_skips_symbol_already_held() -> None:
    print("\nTest: plan does not re-buy a symbol automate_agent already holds")
    existing = [_position("AAPL")]
    candidates = [BoomCandidate("AAPL", "BUY"), BoomCandidate("MSFT", "BUY")]
    plan = plan_automate_trades(candidates, existing, min_positions=1, max_positions=5)
    _assert(plan.to_buy == ["MSFT"], "only buys MSFT, skips already-held AAPL", str(plan.to_buy))


def test_plan_buys_nothing_when_no_buy_candidates() -> None:
    print("\nTest: plan buys nothing and evicts nothing when there are no BUY candidates")
    candidates = [BoomCandidate("AAPL", "HOLD"), BoomCandidate("MSFT", "SELL")]
    plan = plan_automate_trades(candidates, open_positions=[], min_positions=1, max_positions=5)
    _assert(plan.to_buy == [], "no fabricated trades just to hit the min_positions floor")
    _assert(plan.to_evict == [], "no eviction either")


def run_all() -> None:
    print("=" * 60)
    print("AUTOMATE_AGENT DECISION-LOGIC TEST HARNESS")
    print("=" * 60)

    test_rank_keeps_only_buy_decisions()
    test_rank_orders_by_confidence_descending()
    test_rank_tiebreaks_by_predicted_return_pct()
    test_rank_excludes_needs_human_review()
    test_rank_excludes_below_min_confidence()
    test_plan_respects_min_confidence()
    test_confidence_scaled_risk_multiplier_floor_at_low_confidence()
    test_confidence_scaled_risk_multiplier_caps_at_high_confidence()
    test_confidence_scaled_risk_multiplier_interpolates_linearly()
    test_oldest_automate_position_ignores_user_trades()
    test_oldest_automate_position_none_when_all_user_owned()
    test_count_automate_positions()
    test_plan_buys_up_to_available_slots_no_eviction_needed()
    test_plan_never_exceeds_max_positions()
    test_plan_evicts_oldest_when_at_cap()
    test_plan_caps_evictions_per_cycle_even_with_many_fresh_candidates()
    test_plan_never_evicts_a_real_user_position_even_at_cap()
    test_plan_skips_symbol_already_held()
    test_plan_buys_nothing_when_no_buy_candidates()

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    print("=" * 60)
    if FAIL == 0:
        print("\nALL AUTOMATE_AGENT TESTS PASSED")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    run_all()
