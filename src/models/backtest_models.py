"""Pydantic / dataclass models for tastytrade backtest payloads and results."""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Payload models (sent to API)
# ---------------------------------------------------------------------------

@dataclass
class BacktestLeg:
    """A single options leg in the backtest payload.

    type MUST be "equity-option" — any other value returns type=unknown, statistics=null.
    direction MUST be "short" or "long" (not "sell"/"buy").
    Supported strikeSelection values: "strike", "delta", "percentageOtm",
    "priceOffset", and "premium".
    """
    type: str = "equity-option"
    direction: str = "short"
    quantity: int = 1
    side: str = "put"
    days_until_expiration: int = 45
    strike_selection: str = "delta"
    delta: int = 30
    strike_price: Optional[float] = None
    percentage_otm: Optional[float] = None
    price_offset: Optional[float] = None
    premium_amount: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        selection = (self.strike_selection or "delta").strip()
        d = {
            "type": self.type,
            "direction": self.direction,
            "quantity": self.quantity,
            "side": self.side,
            "daysUntilExpiration": self.days_until_expiration,
            "strikeSelection": selection,
        }
        if selection.lower() == "strike" and self.strike_price is not None:
            d["strikeSelection"] = "strike"
            d["strikePrice"] = float(self.strike_price)
        elif selection == "delta":
            d["delta"] = self.delta
        elif selection == "percentageOtm":
            d["percentageOtm"] = self.percentage_otm
        elif selection == "priceOffset":
            d["priceOffset"] = self.price_offset
        elif selection == "premium":
            d["premium"] = self.premium_amount
        else:
            d["delta"] = self.delta
        return d


_EXIT_RULE_MAP: Dict[str, Dict] = {
    "Exit at target date":  {},
    "Exit after N days":    {"type": "days_held"},
    "Target profit 50%":    {"type": "profit_percentage", "profitPercentage": 50},
    "Stop loss 200%":       {"type": "loss_percentage",   "lossPercentage": 200},
}


@dataclass
class BacktestPayload:
    """Full payload sent to POST /backtests."""
    symbol: str
    start_date: str
    end_date: str
    legs: List[BacktestLeg] = field(default_factory=list)
    entry_frequency: str = "every day"
    exit_rule: str = "Exit at target date"
    status: str = "pending"
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    exit_after_days: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        # TastyTrade exitConditions is a Go STRUCT (backtesting.ExitConditions),
        # NOT a slice. Sending a JSON array causes HTTP 400:
        #   "cannot unmarshal array into Go struct field Backtest.exitConditions"
        # Always build a single flat dict — never a list.
        # Include "type" field in all cases (single AND combined) matching proven working format.
        has_tp   = bool(self.take_profit_pct and self.take_profit_pct > 0)
        has_sl   = bool(self.stop_loss_pct   and self.stop_loss_pct   > 0)
        has_days = bool(self.exit_after_days  and self.exit_after_days > 0)

        exit_cond: Dict[str, Any] = {}
        if has_tp:
            exit_cond["profitPercentage"] = float(self.take_profit_pct)
        if has_sl:
            exit_cond["lossPercentage"] = float(self.stop_loss_pct)
        if has_days:
            exit_cond["daysHeld"] = int(self.exit_after_days)

        # Always include type field — use primary condition's type as the struct discriminator.
        # Combined (TP + SL): type="profit_percentage" with both percentage fields present.
        if exit_cond:
            if has_tp:
                exit_cond["type"] = "profit_percentage"
            elif has_sl:
                exit_cond["type"] = "loss_percentage"
            elif has_days:
                exit_cond["type"] = "days_held"

        if not exit_cond:
            exit_cond = _EXIT_RULE_MAP.get(self.exit_rule, {})

        return {
            "startDate": self.start_date,
            "endDate": self.end_date,
            "symbol": self.symbol,
            "status": self.status,
            "entryConditions": {"frequency": self.entry_frequency},
            "exitConditions": exit_cond,
            "legs": [leg.to_dict() for leg in self.legs],
        }


# ---------------------------------------------------------------------------
# Response models (parsed from API)
# ---------------------------------------------------------------------------

@dataclass
class BacktestSnapshot:
    """One price snapshot entry from a backtest trial."""
    date: str
    profit_loss: Decimal
    open_price: Optional[Decimal] = None
    close_price: Optional[Decimal] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BacktestSnapshot":
        def _dec(v: Any) -> Optional[Decimal]:
            try:
                return Decimal(str(v)) if v is not None else None
            except Exception:
                return None

        return cls(
            date=str(d.get("date", "")),
            profit_loss=_dec(d.get("profitLoss") or d.get("profit_loss") or "0") or Decimal("0"),
            open_price=_dec(d.get("openPrice") or d.get("open_price")),
            close_price=_dec(d.get("closePrice") or d.get("close_price")),
        )


@dataclass
class BacktestTrial:
    """One complete trial (open → close) within a backtest."""
    entry_date: str
    exit_date: str
    profit_loss: Decimal
    snapshots: List[BacktestSnapshot] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BacktestTrial":
        def _dec(v: Any) -> Decimal:
            try:
                return Decimal(str(v)) if v is not None else Decimal("0")
            except Exception:
                return Decimal("0")

        snaps = [BacktestSnapshot.from_dict(s) for s in (d.get("snapshots") or [])]
        # API may use openDateTime/closeDateTime or entryDate/exitDate
        entry = (
            d.get("entryDate") or d.get("entry_date")
            or d.get("openDateTime") or d.get("open_date_time") or ""
        )
        exit_ = (
            d.get("exitDate") or d.get("exit_date")
            or d.get("closeDateTime") or d.get("close_date_time") or ""
        )
        return cls(
            entry_date=str(entry),
            exit_date=str(exit_),
            profit_loss=_dec(d.get("profitLoss") or d.get("profit_loss")),
            snapshots=snaps,
            raw=d,
        )


@dataclass
class BacktestStatistics:
    """Aggregate statistics returned by the backtester."""
    total_profit_loss: Decimal
    win_rate: float
    average_profit_loss: Decimal
    num_trades: int
    num_wins: int
    num_losses: int
    max_profit: Decimal
    max_loss: Decimal
    sharpe_ratio: Optional[float] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BacktestStatistics":
        def _dec(v: Any) -> Decimal:
            try:
                return Decimal(str(v)) if v is not None else Decimal("0")
            except Exception:
                return Decimal("0")

        def _f(v: Any, default: float = 0.0) -> float:
            try:
                return float(v) if v is not None else default
            except Exception:
                return default

        def _i(v: Any) -> int:
            try:
                return int(v) if v is not None else 0
            except Exception:
                return 0

        # Handle both camelCase API keys and the human-readable "Win percentage" style
        # that the backtester.vast.tastyworks.com API actually returns
        win_pct_raw = (
            d.get("winRate") or d.get("win_rate")
            or d.get("Win percentage") or d.get("win_percentage") or 0
        )
        # "Win percentage" comes as "36.36" (0-100 scale); winRate comes as 0-1 scale
        win_pct = _f(win_pct_raw)
        if win_pct > 1.0:
            win_pct = win_pct / 100.0

        total_pl_raw = (
            d.get("totalProfitLoss") or d.get("total_profit_loss")
            or d.get("Total profit/loss") or 0
        )
        avg_pl_raw = (
            d.get("averageProfitLoss") or d.get("average_profit_loss")
            or d.get("Avg. profit/loss per trade") or d.get("Avg. return per trade") or 0
        )

        return cls(
            total_profit_loss=_dec(total_pl_raw),
            win_rate=win_pct,
            average_profit_loss=_dec(avg_pl_raw),
            num_trades=_i(
                d.get("numTrades") or d.get("num_trades") or d.get("numberOfTrades")
                or d.get("Number of trades") or 0
            ),
            num_wins=_i(d.get("numWins") or d.get("num_wins") or d.get("Wins") or 0),
            num_losses=_i(d.get("numLosses") or d.get("num_losses") or d.get("Losses") or 0),
            max_profit=_dec(
                d.get("maxProfit") or d.get("max_profit") or d.get("Highest profit") or 0
            ),
            max_loss=_dec(
                d.get("maxLoss") or d.get("max_loss") or d.get("Worst loss") or 0
            ),
            sharpe_ratio=_f(d.get("sharpeRatio") or d.get("sharpe_ratio") or d.get("MAR ratio")) or None,
            raw=d,
        )


@dataclass
class BacktestResult:
    """Complete backtest result including trials and statistics."""
    backtest_id: str
    symbol: str
    status: str
    leg_type: str
    trials: List[BacktestTrial]
    statistics: Optional[BacktestStatistics]
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """True only when the result contains real data (not unknown/null)."""
        if self.leg_type == "unknown":
            return False
        if self.statistics is None:
            return False
        if not self.trials:
            return False
        all_zero = all(t.profit_loss == Decimal("0") for t in self.trials)
        if all_zero and len(self.trials) > 0:
            return False
        return True


@dataclass
class BacktestValidationResult:
    """Validation gate result for a BacktestResult."""
    passed: bool
    reasons: List[str] = field(default_factory=list)
    result: Optional[BacktestResult] = None

    @classmethod
    def fail(cls, *reasons: str) -> "BacktestValidationResult":
        return cls(passed=False, reasons=list(reasons))

    @classmethod
    def ok(cls, result: BacktestResult) -> "BacktestValidationResult":
        return cls(passed=True, reasons=[], result=result)
