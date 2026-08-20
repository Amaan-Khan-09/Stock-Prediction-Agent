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


def rank_boom_candidates(candidates: list[BoomCandidate]) -> list[BoomCandidate]:
    """Keeps only BUY-decision candidates that the prediction engine itself
    didn't flag as needing human review, strongest conviction first.

    needs_human_review is the model's own signal that its call here is
    uncertain enough to want a second opinion. automate_agent is the one
    path in this whole project with no human reviewing before execution --
    silently ignoring that flag specifically here would be the worst place
    to ignore it. Excluding those candidates keeps this consistent with
    the rest of the system's "when the model says it's unsure, a human (or
    here, extra caution) is required" design.
    """
    buys = [
        c for c in candidates
        if c.decision.upper() == "BUY" and not c.needs_human_review
    ]
    return sorted(buys, key=lambda c: (c.confidence, c.predicted_return_pct), reverse=True)


def oldest_automate_position(open_positions: list[dict]) -> Optional[str]:
    """Returns the symbol of the oldest automate_agent-tagged open position,
    or None if there isn't one. Used to free a slot when at the cap -- only
    ever looks at automate_agent's own positions, never a real user's.
    """
    tagged = [
        p for p in open_positions
        if str(p.get("opened_by") or "").lower() == AUTOMATE_AGENT_TAG
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
) -> TradePlan:
    """Decides what to buy and what to evict first, given ranked candidates
    and the currently open automate_agent-tagged positions.

    Never evicts more than necessary, never buys more candidates than exist,
    and never exceeds max_positions total automate_agent-tagged positions.
    Symbols already held (by automate_agent) are skipped -- no point
    "buying" something already open.
    """
    ranked = rank_boom_candidates(candidates)
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
        # ceiling on how many buys this single cycle should attempt.
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
