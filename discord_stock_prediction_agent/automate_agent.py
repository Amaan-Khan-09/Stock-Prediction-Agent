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
(config.automate_agent_watchlist), a hard 1-5 position cap, fixed 1%/10%
stop-loss/take-profit, and every position it opens is tagged so it can
never be confused with (or evict) a position a real user asked for.
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


def rank_boom_candidates(
    candidates: list[BoomCandidate], min_confidence: float = 0.0
) -> list[BoomCandidate]:
    """Keeps only BUY-decision candidates that the prediction engine itself
    didn't flag as needing human review, strongest conviction first.

    needs_human_review is the model's own signal that its call here is
    uncertain enough to want a second opinion. automate_agent is the one
    path in this whole project with no human reviewing before execution --
    silently ignoring that flag specifically here would be the worst place
    to ignore it. Excluding those candidates keeps this consistent with
    the rest of the system's "when the model says it's unsure, a human (or
    here, extra caution) is required" design.

    min_confidence additionally drops BUY calls the model itself didn't flag
    as uncertain, but that still cleared the BUY bar by a thin margin --
    without this, a 51-confidence BUY and a 95-confidence BUY were treated
    identically (aside from sizing/ranking order), which is a lower quality
    bar than the rest of this function's "extra caution" intent implies.
    """
    buys = [
        c for c in candidates
        if c.decision.upper() == "BUY"
        and not c.needs_human_review
        and c.confidence >= min_confidence
    ]
    return sorted(buys, key=lambda c: (c.confidence, c.predicted_return_pct), reverse=True)


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
) -> TradePlan:
    """Decides what to buy and what to evict first, given ranked candidates
    and the currently open automate_agent-tagged positions.

    Never evicts more than necessary, never buys more candidates than exist,
    and never exceeds max_positions total automate_agent-tagged positions.
    Symbols already held (by automate_agent) are skipped -- no point
    "buying" something already open.

    max_evictions_per_cycle additionally caps how much of the existing book
    can be swapped out in one call. Without this, a single cycle where
    max_positions-or-more fresh candidates all outrank the current holdings
    could evict the entire book at once -- a lot of turnover from one
    scan, and inconsistent with this module's stated "bounded on purpose to
    limit risk" design. The default of 1 means at most one position rotates
    per cycle; a full book gradually rotates across multiple cycles instead.
    """
    ranked = rank_boom_candidates(candidates, min_confidence)
    held_symbols = {
        str(p.get("symbol") or "").upper()
        for p in open_positions
        if str(p.get("opened_by") or "").lower() == AUTOMATE_AGENT_TAG
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
