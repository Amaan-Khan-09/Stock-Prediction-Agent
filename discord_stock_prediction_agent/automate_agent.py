"""Pure decision logic for the `!automate_agent` autonomous scan-and-trade
command.

Deliberately split from discord_agent.py's async orchestration (which does
the real Alpaca/Gemini calls) so the actual *decisions* -- which candidate
to buy, which existing position to evict when at the slot cap -- are plain,
synchronous, fully unit-testable functions with no network calls, mirroring
how the rest of this codebase keeps decision logic separate from I/O.

Unlike every other trade path in this agent, `!automate_agent` is a
deliberate, explicitly-requested exception to "a human originates every
signal": the human trigger is the command itself, not a specific ticker.
The scope is bounded on purpose to limit that risk: a fixed watchlist
(config.automate_agent_watchlist), a hard position cap
(config.automate_agent_max_positions), fixed stop-loss/take-profit (equity:
config.equity_stop_loss_pct/take_profit_pct; automate_agent's own options:
config.automate_agent_option_stop_loss_pct/take_profit_pct, a wider
premium-based band since premium swings far more than the underlying), and
every position it opens is tagged so it can never be confused with (or
evict) a position a real user asked for.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

AUTOMATE_AGENT_TAG = "automate_agent"


@dataclass(frozen=True)
class BoomCandidate:
    symbol: str
    decision: str  # "BUY" | "SELL" | "HOLD" | "REVIEW" from run_project_prediction
    confidence: float = 0.0  # 0-100 confidence_score from the prediction engine
    predicted_return_pct: float = 0.0  # used as a tiebreaker when confidence ties
    needs_human_review: bool = False  # the model's OWN flag that this call is uncertain
    # The same absolute price target that produced predicted_return_pct (and
    # therefore this BUY/SELL decision in the first place). None if the
    # prediction result didn't carry one -- callers must degrade gracefully,
    # not assume it's always present.
    predicted_target_price: Optional[float] = None


def rank_boom_candidates(
    candidates: list[BoomCandidate], min_confidence: float = 0.0, include_sell: bool = False
) -> list[BoomCandidate]:
    """Keeps only BUY-decision candidates (and, if include_sell, SELL-decision
    ones too) that the prediction engine itself didn't flag as needing human
    review, strongest conviction first.

    include_sell exists for the options path: a SELL call is just as
    actionable as a BUY one when the position itself is a put (profiting
    from an expected decline) rather than a long share position -- equity
    callers leave this False since automate_agent never shorts stock.

    needs_human_review is the model's own signal that its call here is
    uncertain enough to want a second opinion. automate_agent is the one
    path in this whole project with no human reviewing before execution --
    silently ignoring that flag specifically here would be the worst place
    to ignore it. Excluding those candidates keeps this consistent with
    the rest of the system's "when the model says it's unsure, a human (or
    here, extra caution) is required" design.

    min_confidence additionally drops calls the model itself didn't flag
    as uncertain, but that still cleared the bar by a thin margin --
    without this, a 51-confidence call and a 95-confidence call were treated
    identically (aside from sizing/ranking order), which is a lower quality
    bar than the rest of this function's "extra caution" intent implies.
    """
    decisions = {"BUY", "SELL"} if include_sell else {"BUY"}
    matches = [
        c for c in candidates
        if c.decision.upper() in decisions
        and not c.needs_human_review
        and c.confidence >= min_confidence
    ]
    # Tiebreak by conviction *in the direction of this candidate's own
    # decision* -- a BUY's predicted_return_pct is normally positive and a
    # SELL's normally negative, so ranking by the raw signed value would
    # silently bias every confidence-tie toward BUY candidates regardless
    # of which direction's move is actually predicted to be larger. This
    # is deliberately not just abs(): a plain abs() trusts that sign and
    # decision always agree, which isn't actually guaranteed upstream (a
    # decision can be adjusted by a downstream override without
    # necessarily recomputing predicted_return_pct to match) -- flipping
    # SELL's sign relative to its own decision, rather than discarding
    # sign entirely, means a decision/return-sign mismatch demotes a
    # candidate instead of inflating it. When include_sell is False (the
    # equity path, unchanged), every match is a BUY, so this reduces to
    # the original signed value -- a no-op relative to prior behavior.
    def _conviction(c: BoomCandidate) -> float:
        return c.predicted_return_pct if c.decision.upper() == "BUY" else -c.predicted_return_pct

    return sorted(matches, key=lambda c: (c.confidence, _conviction(c)), reverse=True)


def confidence_scaled_risk_multiplier(confidence: float, floor: float = 0.75, cap: float = 1.25) -> float:
    """Scales a candidate's position size by model conviction, within a
    deliberately narrow +/-25% band -- not full Kelly sizing (too aggressive
    for the one execution path in this whole project with no human review),
    just a conservative tilt toward higher-confidence picks. Comparable
    autonomous trading agents size this way too (e.g. a documented pattern
    of +15% position size above 85% confidence) rather than treating every
    BUY the same regardless of how sure the model actually is.

    confidence is on run_project_prediction's native 0-100 scale. Linearly
    interpolates between floor (at confidence<=50) and cap (at
    confidence>=100); clamped outside that range so a malformed/negative
    confidence value can never blow past the band.
    """
    if confidence <= 50:
        return floor
    if confidence >= 100:
        return cap
    frac = (confidence - 50) / 50.0
    return floor + frac * (cap - floor)


@dataclass(frozen=True)
class StrikeQuote:
    occ_symbol: str
    strike: float
    premium: float  # per-share option premium; one contract costs premium * 100


def rank_strikes(
    quotes: list[StrikeQuote],
    side: str,
    predicted_target_price: Optional[float],
    current_price: float,
    risk_budget: float,
) -> list[StrikeQuote]:
    """Ranks which listed strikes (from a handful of quoted candidates near
    the money) are actually the best trade, best first, instead of always
    taking the nearest-the-money strike regardless of what the model
    itself expects.

    There is still no options-greeks/delta data source anywhere in this
    codebase, so this can't rank by real delta/theta. What IS already
    available is predicted_target_price -- the same absolute price target
    that produced the BUY/SELL decision for this symbol in the first
    place. Scoring each candidate strike by its expected payoff if that
    target is reached (intrinsic value at the target, minus the premium
    paid, as a fraction of that premium) reuses a signal this codebase
    already computed and validated the decision on, rather than inventing
    a fresh, unvalidated one.

    Strikes whose premium doesn't fit risk_budget are dropped before
    scoring -- a strike this trade literally can't afford isn't "the best
    trade," it isn't a trade at all. Returns [] if nothing quoted fits.

    Falls back to ranking by nearest-the-money among what's affordable
    when predicted_target_price isn't available (matches the prior,
    simpler behavior); ties in expected payoff also break toward nearest-
    the-money, so this converges to the old ATM-only ranking as the
    model's own predicted move shrinks toward zero.

    The ranked (not just single-best) list exists because the payoff
    heuristic here is still just that -- a heuristic, not a guarantee. The
    real backtest gate (run_options_strategy_validation) can legitimately
    disagree with strike #1; giving it #2 and #3 to consider too, instead
    of only ever offering its single top pick, means a real disagreement
    between the heuristic and the backtest doesn't have to end the cycle
    with no trade at all.
    """
    affordable = [q for q in quotes if q.premium > 0 and q.premium * 100.0 <= risk_budget]
    if not affordable:
        return []
    if predicted_target_price is None or predicted_target_price <= 0:
        return sorted(affordable, key=lambda q: abs(q.strike - current_price))

    is_call = side.upper() == "CALL"

    def expected_profit_ratio(q: StrikeQuote) -> float:
        intrinsic_at_target = (
            max(0.0, predicted_target_price - q.strike) if is_call
            else max(0.0, q.strike - predicted_target_price)
        )
        return (intrinsic_at_target - q.premium) / q.premium

    return sorted(
        affordable,
        key=lambda q: (expected_profit_ratio(q), -abs(q.strike - current_price)),
        reverse=True,
    )


def select_best_strike(
    quotes: list[StrikeQuote],
    side: str,
    predicted_target_price: Optional[float],
    current_price: float,
    risk_budget: float,
) -> Optional[StrikeQuote]:
    """The single top-ranked strike, or None if nothing quoted is
    affordable. See rank_strikes for the actual ranking logic; kept as its
    own function since most callers (and every existing test) only need
    the single best pick, not the full ranking.
    """
    ranked = rank_strikes(quotes, side, predicted_target_price, current_price, risk_budget)
    return ranked[0] if ranked else None


def oldest_automate_position(open_positions: list[dict]) -> Optional[str]:
    """Returns the symbol of the oldest automate_agent-tagged open position,
    or None if there isn't one. Used to free a slot when at the cap -- only
    ever looks at automate_agent's own positions, never a real user's.

    Only ever selects an equity position: the eviction loop that acts on
    this result can only sell-to-close equity today (there is no option
    eviction path yet), so selecting an option position here would schedule
    an eviction that silently never happens -- the "freed" slot never
    actually frees, while the buy that assumed it would still proceeds,
    letting the real position count exceed the intended cap. Option
    positions are still counted by count_automate_positions; they just
    can't be picked as the thing to evict.
    """
    tagged = [
        p for p in open_positions
        if str(p.get("opened_by") or "").lower() == AUTOMATE_AGENT_TAG
        and str(p.get("asset_type") or "equity").lower() != "option"
    ]
    if not tagged:
        return None
    tagged.sort(key=lambda p: str(p.get("updated_at") or ""))
    symbol = tagged[0].get("symbol")
    return str(symbol).upper() if symbol else None


def count_automate_positions(open_positions: list[dict]) -> int:
    return sum(
        1 for p in open_positions
        if str(p.get("opened_by") or "").lower() == AUTOMATE_AGENT_TAG
    )


@dataclass(frozen=True)
class TradePlan:
    to_evict: list[str]  # symbols to sell first, to free slots
    to_buy: list[str]  # symbols to buy, in order


def plan_automate_trades(
    candidates: list[BoomCandidate],
    open_positions: list[dict],
    min_positions: int,
    max_positions: int,
    min_confidence: float = 0.0,
    max_evictions_per_cycle: int = 1,
    include_sell: bool = False,
    asset_type: str = "equity",
) -> TradePlan:
    """Decides what to buy and what to evict first, given ranked candidates
    and the currently open automate_agent-tagged positions.

    Never evicts more than necessary, never buys more candidates than exist,
    and never exceeds max_positions total automate_agent-tagged positions.
    Symbols already held (by automate_agent) *in this same asset_type* are
    skipped -- no point "buying" something already open.

    asset_type scopes that "already held" check to the asset class this
    plan is being computed for. open_positions is expected to be the full
    combined equity+option book (asset-tagged via each dict's own
    "asset_type" key, equity positions defaulting to "equity" when absent)
    so max_positions still counts both together -- but a symbol held as
    equity must not by itself block that same symbol from also being
    bought as an option, and vice versa. "both" mode's whole point is
    holding both at once for one symbol; without this, whichever leg
    fills first would silently lock the other out of ever completing.

    max_evictions_per_cycle additionally caps how much of the existing book
    can be swapped out in one call. Without this, a single cycle where
    max_positions-or-more fresh candidates all outrank the current holdings
    could evict the entire book at once -- a lot of turnover from one
    scan, and inconsistent with this module's stated "bounded on purpose to
    limit risk" design. The default of 1 means at most one position rotates
    per cycle; a full book gradually rotates across multiple cycles instead.
    """
    ranked = rank_boom_candidates(candidates, min_confidence, include_sell)
    held_symbols = {
        str(p.get("symbol") or "").upper()
        for p in open_positions
        if str(p.get("opened_by") or "").lower() == AUTOMATE_AGENT_TAG
        and str(p.get("asset_type") or "equity").lower() == asset_type.lower()
    }
    fresh = [c.symbol.upper() for c in ranked if c.symbol.upper() not in held_symbols]
    if not fresh:
        return TradePlan(to_evict=[], to_buy=[])

    current_count = count_automate_positions(open_positions)
    available_slots = max(0, max_positions - current_count)
    to_buy: list[str] = []
    to_evict: list[str] = []

    remaining_positions = list(open_positions)
    for symbol in fresh:
        if len(to_buy) >= max_positions:
            break
        if available_slots > 0:
            available_slots -= 1
            to_buy.append(symbol)
            continue
        # At cap -- evict the oldest automate_agent position to make room,
        # but only if we haven't already hit the overall max_positions
        # ceiling on how many buys this single cycle should attempt, or the
        # per-cycle eviction limit.
        if len(to_evict) >= max_evictions_per_cycle:
            break
        evict = oldest_automate_position(remaining_positions)
        if evict is None:
            break
        to_evict.append(evict)
        remaining_positions = [p for p in remaining_positions if p.get("symbol") != evict]
        to_buy.append(symbol)

    # min_positions is a floor on *ambition*, not a guarantee -- if fewer
    # than min_positions genuine BUY candidates exist, we still only ever
    # buy what was actually found. Never fabricate a trade to hit a quota.
    del min_positions
    return TradePlan(to_evict=to_evict, to_buy=to_buy)
