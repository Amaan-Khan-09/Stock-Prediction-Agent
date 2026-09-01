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
    StrikeQuote,
    confidence_scaled_risk_multiplier,
    count_automate_positions,
    oldest_automate_position,
    plan_automate_trades,
    rank_boom_candidates,
    rank_strikes,
    select_best_strike,
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


def test_rank_include_sell_false_still_excludes_sell() -> None:
    print("\nTest: include_sell defaults False -- equity path behavior is unchanged")
    candidates = [BoomCandidate("AAPL", "BUY"), BoomCandidate("NVDA", "SELL")]
    ranked = rank_boom_candidates(candidates)
    _assert([c.symbol for c in ranked] == ["AAPL"], "SELL still excluded when include_sell isn't passed", str(ranked))


def test_rank_include_sell_true_keeps_sell_decisions_too() -> None:
    print("\nTest: include_sell=True keeps SELL-decision candidates alongside BUY ones")
    candidates = [
        BoomCandidate("AAPL", "BUY", confidence=80),
        BoomCandidate("NVDA", "SELL", confidence=90),
        BoomCandidate("MSFT", "HOLD", confidence=99),
        BoomCandidate("TSLA", "REVIEW", confidence=99),
    ]
    ranked = rank_boom_candidates(candidates, include_sell=True)
    _assert(
        [c.symbol for c in ranked] == ["NVDA", "AAPL"],
        "both BUY and SELL kept (strongest confidence first), HOLD/REVIEW still excluded",
        str(ranked),
    )


def test_rank_include_sell_tiebreak_uses_abs_predicted_return() -> None:
    print("\nTest: include_sell tiebreak compares |predicted_return_pct|, not the signed value")
    candidates = [
        # A SELL's predicted_return_pct is naturally negative (predicting a
        # decline) -- without abs(), this would always lose the tiebreak to
        # any BUY at the same confidence regardless of move size.
        BoomCandidate("NVDA", "SELL", confidence=70, predicted_return_pct=-5.0),
        BoomCandidate("AAPL", "BUY", confidence=70, predicted_return_pct=2.0),
    ]
    ranked = rank_boom_candidates(candidates, include_sell=True)
    _assert(
        [c.symbol for c in ranked] == ["NVDA", "AAPL"],
        "NVDA's larger-magnitude -5% move outranks AAPL's smaller +2% move at equal confidence",
        str(ranked),
    )


def test_plan_include_sell_plans_a_sell_candidate_as_a_buy() -> None:
    print("\nTest: plan_automate_trades' include_sell surfaces a SELL candidate as something to buy (a put)")
    candidates = [BoomCandidate("TSLA", "SELL", confidence=80)]
    plan = plan_automate_trades(
        candidates, open_positions=[], min_positions=1, max_positions=5, include_sell=True
    )
    _assert(plan.to_buy == ["TSLA"], "the SELL-decision symbol is queued to buy (as a put)", str(plan.to_buy))


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


def test_select_best_strike_falls_back_to_nearest_the_money_without_a_target() -> None:
    print("\nTest: select_best_strike falls back to nearest-the-money when there's no predicted target")
    quotes = [
        StrikeQuote("A95", 95.0, premium=3.0),
        StrikeQuote("A100", 100.0, premium=2.0),
        StrikeQuote("A105", 105.0, premium=1.0),
    ]
    chosen = select_best_strike(quotes, "CALL", None, current_price=101.0, risk_budget=1000.0)
    _assert(chosen is not None and chosen.strike == 100.0, "picks the strike nearest the $101 spot", chosen)


def test_select_best_strike_picks_highest_expected_payoff_for_a_call() -> None:
    print("\nTest: select_best_strike picks the CALL strike with the best expected payoff at the predicted target")
    quotes = [
        StrikeQuote("A95", 95.0, premium=7.0),   # intrinsic@110=15, ratio=1.14
        StrikeQuote("A100", 100.0, premium=4.0),  # intrinsic@110=10, ratio=1.5
        StrikeQuote("A105", 105.0, premium=1.0),  # intrinsic@110=5,  ratio=4.0 <- best
    ]
    chosen = select_best_strike(quotes, "CALL", 110.0, current_price=100.0, risk_budget=1000.0)
    _assert(
        chosen is not None and chosen.strike == 105.0,
        "the cheap OTM call has the best expected payoff, not the ATM one",
        chosen,
    )


def test_rank_strikes_returns_the_full_order_not_just_the_winner() -> None:
    print("\nTest: rank_strikes returns every affordable strike ranked best-to-worst, not only the top pick")
    quotes = [
        StrikeQuote("A95", 95.0, premium=7.0),    # ratio=1.14 -- 3rd
        StrikeQuote("A100", 100.0, premium=4.0),  # ratio=1.5  -- 2nd
        StrikeQuote("A105", 105.0, premium=1.0),  # ratio=4.0  -- 1st
    ]
    ranked = rank_strikes(quotes, "CALL", 110.0, current_price=100.0, risk_budget=1000.0)
    _assert(
        [q.strike for q in ranked] == [105.0, 100.0, 95.0],
        "full ranking, best expected payoff first",
        [q.strike for q in ranked],
    )
    _assert(
        select_best_strike(quotes, "CALL", 110.0, current_price=100.0, risk_budget=1000.0) is ranked[0],
        "select_best_strike is just rank_strikes()[0]",
    )


def test_rank_strikes_drops_unaffordable_strikes_from_the_ranking() -> None:
    print("\nTest: rank_strikes excludes strikes the risk budget can't afford, same as select_best_strike")
    quotes = [
        StrikeQuote("Cheap", 105.0, premium=1.0),   # cost $100, affordable
        StrikeQuote("Pricey", 90.0, premium=12.0),  # cost $1200, not affordable at $500 budget
    ]
    ranked = rank_strikes(quotes, "CALL", 110.0, current_price=100.0, risk_budget=500.0)
    _assert([q.strike for q in ranked] == [105.0], "only the affordable strike survives", ranked)


def test_rank_strikes_returns_empty_list_when_nothing_affordable() -> None:
    print("\nTest: rank_strikes returns [] (not None, not a crash) when nothing quoted is affordable")
    quotes = [StrikeQuote("A100", 100.0, premium=10.0)]
    ranked = rank_strikes(quotes, "CALL", 110.0, current_price=100.0, risk_budget=500.0)
    _assert(ranked == [], "empty list when the cheapest strike still exceeds the budget", ranked)


def test_select_best_strike_picks_highest_expected_payoff_for_a_put() -> None:
    print("\nTest: select_best_strike mirrors the same logic for PUTs")
    quotes = [
        StrikeQuote("A105", 105.0, premium=7.0),  # intrinsic@90=15, ratio=1.14
        StrikeQuote("A100", 100.0, premium=4.0),  # intrinsic@90=10, ratio=1.5
        StrikeQuote("A95", 95.0, premium=1.0),    # intrinsic@90=5,  ratio=4.0 <- best
    ]
    chosen = select_best_strike(quotes, "PUT", 90.0, current_price=100.0, risk_budget=1000.0)
    _assert(
        chosen is not None and chosen.strike == 95.0,
        "the cheap OTM put has the best expected payoff, not the ATM one",
        chosen,
    )


def test_select_best_strike_drops_strikes_the_risk_budget_cannot_afford() -> None:
    print("\nTest: select_best_strike never returns a strike whose premium exceeds risk_budget")
    quotes = [
        StrikeQuote("DeepITM", 90.0, premium=12.0),  # intrinsic@101=11, ratio=-0.083 (best if affordable)
        StrikeQuote("ATM", 100.0, premium=3.0),       # intrinsic@101=1,  ratio=-0.667
        StrikeQuote("OTM", 110.0, premium=0.5),       # intrinsic@101=0,  ratio=-1.0
    ]
    tight = select_best_strike(quotes, "CALL", 101.0, current_price=100.0, risk_budget=500.0)
    _assert(
        tight is not None and tight.strike == 100.0,
        "DeepITM (cost $1200) doesn't fit a $500 budget, so the next-best affordable strike wins",
        tight,
    )
    loose = select_best_strike(quotes, "CALL", 101.0, current_price=100.0, risk_budget=2000.0)
    _assert(
        loose is not None and loose.strike == 90.0,
        "with enough budget to afford it, DeepITM's better expected payoff wins",
        loose,
    )


def test_select_best_strike_returns_none_when_nothing_affordable() -> None:
    print("\nTest: select_best_strike returns None rather than a trade the risk budget can't support")
    quotes = [StrikeQuote("A100", 100.0, premium=10.0)]
    chosen = select_best_strike(quotes, "CALL", 110.0, current_price=100.0, risk_budget=500.0)
    _assert(chosen is None, "no strike fits a $500 budget when the cheapest costs $1000/contract", chosen)


def test_oldest_automate_position_ignores_user_trades() -> None:
    print("\nTest: eviction never targets a real user's position")
    positions = [
        _position("AAPL", opened_by="", updated_at="2026-01-01T00:00:00Z"),  # user trade, oldest
        _position("MSFT", opened_by=AUTOMATE_AGENT_TAG, updated_at="2026-01-02T00:00:00Z"),
        _position("NVDA", opened_by=AUTOMATE_AGENT_TAG, updated_at="2026-01-01T12:00:00Z"),
    ]
    evict = oldest_automate_position(positions)
    _assert(evict == "NVDA", "picks the oldest automate_agent position, not the user's older AAPL", str(evict))


def test_oldest_automate_position_never_selects_an_option_position() -> None:
    print("\nTest: eviction never picks an option position even if it's the oldest overall")
    # Regression: the eviction loop that acts on this result can only
    # sell-to-close equity today. If this picked the oldest option position,
    # plan_automate_trades would schedule an eviction that silently never
    # happens (the equity-only eviction loop can't find a matching equity
    # position to sell), while the buy that assumed the slot was freed still
    # proceeds -- letting the real position count exceed the cap.
    positions = [
        {"symbol": "TSLA", "opened_by": AUTOMATE_AGENT_TAG, "updated_at": "2026-01-01T00:00:00Z", "qty": 1, "asset_type": "option"},
        _position("MSFT", updated_at="2026-01-02T00:00:00Z"),
        _position("NVDA", updated_at="2026-01-03T00:00:00Z"),
    ]
    evict = oldest_automate_position(positions)
    _assert(
        evict == "MSFT",
        "skips the older TSLA option position, picks the oldest equity position (MSFT) instead",
        str(evict),
    )


def test_oldest_automate_position_none_when_only_options_are_held() -> None:
    print("\nTest: eviction returns None (never forces a bad pick) when only option positions exist")
    positions = [
        {"symbol": "TSLA", "opened_by": AUTOMATE_AGENT_TAG, "updated_at": "2026-01-01T00:00:00Z", "qty": 1, "asset_type": "option"},
    ]
    evict = oldest_automate_position(positions)
    _assert(evict is None, "no equity candidate exists to evict, so None is correct -- not a wrong pick")


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


def test_plan_asset_type_scopes_the_already_held_check() -> None:
    print("\nTest: asset_type scopes 'already held' so an equity position doesn't block the option plan for the same symbol")
    existing = [_position("AAPL")]  # held as equity -- _position's asset_type defaults to "equity"
    candidates = [BoomCandidate("AAPL", "BUY")]
    equity_plan = plan_automate_trades(candidates, existing, min_positions=1, max_positions=5, asset_type="equity")
    option_plan = plan_automate_trades(candidates, existing, min_positions=1, max_positions=5, asset_type="option")
    _assert(
        equity_plan.to_buy == [],
        "AAPL is already held as equity -- equity's own plan must skip it",
        str(equity_plan.to_buy),
    )
    _assert(
        option_plan.to_buy == ["AAPL"],
        "AAPL is NOT held as an option -- the option plan must still buy it, not treat it as already-held",
        str(option_plan.to_buy),
    )


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
    test_rank_include_sell_false_still_excludes_sell()
    test_rank_include_sell_true_keeps_sell_decisions_too()
    test_rank_include_sell_tiebreak_uses_abs_predicted_return()
    test_plan_include_sell_plans_a_sell_candidate_as_a_buy()
    test_plan_respects_min_confidence()
    test_confidence_scaled_risk_multiplier_floor_at_low_confidence()
    test_confidence_scaled_risk_multiplier_caps_at_high_confidence()
    test_confidence_scaled_risk_multiplier_interpolates_linearly()
    test_select_best_strike_falls_back_to_nearest_the_money_without_a_target()
    test_select_best_strike_picks_highest_expected_payoff_for_a_call()
    test_select_best_strike_picks_highest_expected_payoff_for_a_put()
    test_select_best_strike_drops_strikes_the_risk_budget_cannot_afford()
    test_select_best_strike_returns_none_when_nothing_affordable()
    test_rank_strikes_returns_the_full_order_not_just_the_winner()
    test_rank_strikes_drops_unaffordable_strikes_from_the_ranking()
    test_rank_strikes_returns_empty_list_when_nothing_affordable()
    test_oldest_automate_position_ignores_user_trades()
    test_oldest_automate_position_none_when_all_user_owned()
    test_count_automate_positions()
    test_plan_buys_up_to_available_slots_no_eviction_needed()
    test_plan_never_exceeds_max_positions()
    test_plan_evicts_oldest_when_at_cap()
    test_plan_caps_evictions_per_cycle_even_with_many_fresh_candidates()
    test_plan_never_evicts_a_real_user_position_even_at_cap()
    test_plan_skips_symbol_already_held()
    test_plan_asset_type_scopes_the_already_held_check()
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
