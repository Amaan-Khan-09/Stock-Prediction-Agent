"""Options contract models, enums, and validation state types.

Used by: discord_parser, options_decision_gate, signal_review_service,
          tastytrade services, and Streamlit UI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Constants — contract selection methods
# ──────────────────────────────────────────────────────────────────────────────

class ContractMethod:
    EXACT_STRIKE       = "EXACT_STRIKE"
    DELTA_SELECTION    = "DELTA_SELECTION"
    DELTA_PROXY        = "DELTA_PROXY_APPROXIMATE"


# ──────────────────────────────────────────────────────────────────────────────
# Constants — validation / order statuses
# ──────────────────────────────────────────────────────────────────────────────

class VS:  # ValidationStatus
    SUCCESS          = "SUCCESS"
    FAILED           = "FAILED"
    SKIPPED          = "SKIPPED"
    NOT_REQUESTED    = "NOT_REQUESTED"
    NOT_CONFIGURED   = "NOT_CONFIGURED"
    NOT_AVAILABLE    = "NOT_AVAILABLE"
    REVIEW_REQUIRED  = "REVIEW_REQUIRED"
    NO_TRADE         = "NO_TRADE"
    SUBMITTED        = "SUBMITTED"
    NOT_SUBMITTED    = "NOT_SUBMITTED"
    DEFAULT_ASSUMED  = "DEFAULT_ASSUMED"
    MISSING          = "MISSING"
    NOT_PROVIDED     = "NOT_PROVIDED"


# ──────────────────────────────────────────────────────────────────────────────
# Constants — asset / signal types
# ──────────────────────────────────────────────────────────────────────────────

class AssetType:
    STOCK   = "STOCK"
    OPTION  = "OPTION"
    UNKNOWN = "UNKNOWN"


class SignalAction:
    BUY     = "BUY"
    SELL    = "SELL"
    EXIT    = "EXIT"
    TRIM    = "TRIM"
    HOLD    = "HOLD"
    WATCH   = "WATCH"
    UNKNOWN = "UNKNOWN"


class OptionSide:
    CALL = "CALL"
    PUT  = "PUT"


class SignalStatus:
    OPEN    = "OPEN"
    PARTIAL = "PARTIAL"
    CLOSED  = "CLOSED"
    REVIEW  = "REVIEW"
    INVALID = "INVALID"


# ──────────────────────────────────────────────────────────────────────────────
# Parsed Signal
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ParsedSignal:
    """Result of parsing a raw trading message (stock or option)."""
    raw_message: str

    asset_type:  str = AssetType.UNKNOWN
    action:      str = SignalAction.UNKNOWN

    # Common
    underlying: Optional[str] = None

    # Options-specific
    option_type:  Optional[str]   = None    # CALL / PUT
    strike:       Optional[float] = None    # e.g. 620.0, 7570.0 — NO 0-99 cap
    expiry_date:  Optional[str]   = None    # YYYY-MM-DD
    dte:          Optional[int]   = None
    premium:      Optional[float] = None
    contracts:    Optional[int]   = None

    # Stock-specific
    quantity: Optional[int] = None

    # Delta-based selection
    delta_ui:      Optional[int]   = None   # 0-100 UI scale
    delta_decimal: Optional[float] = None   # 0.00-1.00 internal

    # Parser metadata
    normalized_message:     str   = ""
    parser_confidence:      float = 0.0
    missing_fields:         List[str] = field(default_factory=list)
    reply_to_message_id:    Optional[str] = None
    linked_signal_id:       Optional[str] = None
    high_risk_tag:          bool  = False
    contract_selection_method: str = ContractMethod.DELTA_SELECTION
    raw_extras:             Dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Options Contract Request
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class OptionsContractRequest:
    """Requested options contract extracted from a signal."""
    underlying:   Optional[str]   = None
    option_type:  Optional[str]   = None    # CALL / PUT
    strike:       Optional[float] = None
    expiry_date:  Optional[str]   = None
    dte:          Optional[int]   = None
    premium:      Optional[float] = None
    contracts:    int             = 1
    side:         Optional[str]   = None    # BUY / SELL
    contract_selection_method: str = ContractMethod.DELTA_SELECTION
    delta_ui:      Optional[int]   = None
    delta_decimal: Optional[float] = None
    occ_symbol:   Optional[str]   = None

    def build_occ_symbol(self) -> Optional[str]:
        """Build OCC symbol if all required fields are available.
        Example: SPY + 2026-07-17 + CALL + 620 -> SPY260717C00620000
        """
        if not all([self.underlying, self.option_type, self.strike is not None, self.expiry_date]):
            return None
        try:
            from datetime import datetime
            dt = datetime.strptime(self.expiry_date, "%Y-%m-%d")
            date_str = dt.strftime("%y%m%d")
            cp = "C" if self.option_type == OptionSide.CALL else "P"
            strike_int = int(round(self.strike * 1000))
            return f"{self.underlying.upper()}{date_str}{cp}{strike_int:08d}"
        except Exception:
            return None


# ──────────────────────────────────────────────────────────────────────────────
# Options Validation State
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class OptionsValidationState:
    """Complete state of all validation steps for one options signal."""

    # --- What was requested ---
    requested_underlying:   Optional[str]   = None
    requested_option_type:  Optional[str]   = None
    requested_strike:       Optional[float] = None
    requested_expiry:       Optional[str]   = None
    requested_dte:          Optional[int]   = None
    requested_premium:      Optional[float] = None
    requested_contracts:    int             = 1
    contract_selection_method: str          = ContractMethod.DELTA_SELECTION

    # --- Delta info ---
    delta_ui:      Optional[int]   = None
    delta_decimal: Optional[float] = None

    # --- Validation statuses ---
    exact_validation_status:  str  = VS.NOT_REQUESTED
    delta_proxy_status:       str  = VS.NOT_REQUESTED
    validation_type:          str  = ""
    fallback_used:            bool = False
    eligible_for_exact_accuracy: bool = False

    # --- Contract existence ---
    exact_contract_found:     Optional[bool]  = None
    alpaca_contract_symbol:   Optional[str]   = None
    occ_symbol:               Optional[str]   = None
    option_chain_status:      str             = VS.NOT_CONFIGURED
    nearest_available_strikes: List[float]    = field(default_factory=list)
    strike_distance_pct:      Optional[float] = None

    # --- Tastytrade backtest ---
    tastytrade_status:           str            = VS.NOT_CONFIGURED
    options_backtest_decision:   Optional[str]  = None
    options_backtest_pnl:        Optional[float]= None
    options_backtest_win_rate:   Optional[float]= None
    options_backtest_raw:        Optional[dict] = None

    # --- AI stock prediction context (supporting info only) ---
    ai_stock_decision:    Optional[str]   = None
    ai_stock_return_pct:  Optional[float] = None
    ai_confidence:        Optional[int]   = None
    ai_risk:              Optional[int]   = None

    # --- Final decision ---
    final_agent_output:   str  = VS.REVIEW_REQUIRED
    order_status:         str  = VS.NOT_SUBMITTED
    manual_override_used: bool = False
    reason:               str  = ""

    def to_display_dict(self) -> Dict[str, Any]:
        """Return a flat dict suitable for Streamlit/Discord display.
        Every field that is None is shown as NOT_PROVIDED.
        """
        def _fmt(v: Any) -> str:
            if v is None:
                return VS.NOT_PROVIDED
            if isinstance(v, bool):
                return str(v).upper()
            if isinstance(v, float):
                return f"{v:.4f}" if abs(v) < 10 else f"{v:.2f}"
            return str(v)

        return {
            "Requested Underlying":      _fmt(self.requested_underlying),
            "Requested Option Type":     _fmt(self.requested_option_type),
            "Requested Strike":          _fmt(self.requested_strike),
            "Requested Expiry":          _fmt(self.requested_expiry),
            "Requested DTE":             _fmt(self.requested_dte),
            "Requested Premium":         _fmt(self.requested_premium),
            "Requested Contracts":       _fmt(self.requested_contracts),
            "Contract Selection Method": self.contract_selection_method,
            "Delta UI":                  _fmt(self.delta_ui),
            "Delta Decimal":             _fmt(self.delta_decimal),
            "Exact Strike Validation":   self.exact_validation_status,
            "Delta Proxy Validation":    self.delta_proxy_status,
            "Validation Type":           self.validation_type or VS.NOT_PROVIDED,
            "Fallback Used":             _fmt(self.fallback_used),
            "Eligible For Exact Accuracy": _fmt(self.eligible_for_exact_accuracy),
            "OCC Symbol":                _fmt(self.occ_symbol),
            "Exact Contract Found":      _fmt(self.exact_contract_found),
            "Option Chain Status":       self.option_chain_status,
            "Nearest Strikes":           str(self.nearest_available_strikes) if self.nearest_available_strikes else VS.NOT_PROVIDED,
            "Strike Distance %":         _fmt(self.strike_distance_pct),
            "Tastytrade Status":         self.tastytrade_status,
            "Options Backtest Decision": _fmt(self.options_backtest_decision),
            "Options Backtest P&L":      _fmt(self.options_backtest_pnl),
            "Options Win Rate":          _fmt(self.options_backtest_win_rate),
            "AI Stock Decision":         _fmt(self.ai_stock_decision),
            "AI Return %":               _fmt(self.ai_stock_return_pct),
            "AI Confidence":             _fmt(self.ai_confidence),
            "AI Risk":                   _fmt(self.ai_risk),
            "Final Agent Output":        self.final_agent_output,
            "Order Status":              self.order_status,
            "Manual Override Used":      _fmt(self.manual_override_used),
            "Reason":                    self.reason or VS.NOT_PROVIDED,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Context Trade (same-day reply chain)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ContextTrade:
    """A linked trade record tracking entry, partials, and full exit."""
    trade_id:           str
    date:               str
    message_id:         Optional[str]  = None
    reply_to:           Optional[str]  = None
    user_label:         Optional[str]  = None

    underlying:   Optional[str]   = None
    asset_type:   str             = AssetType.UNKNOWN
    option_type:  Optional[str]   = None
    strike:       Optional[float] = None
    expiry:       Optional[str]   = None

    entry_price:       Optional[float] = None
    contracts_opened:  int = 0
    contracts_closed:  int = 0
    remaining_contracts: int = 0

    exit_prices:   List[float] = field(default_factory=list)
    exit_contracts: List[int]  = field(default_factory=list)

    realized_pnl:  Optional[float] = None
    status:        str = SignalStatus.OPEN

    raw_messages:  List[str] = field(default_factory=list)

    def compute_pnl(self) -> Optional[float]:
        """Compute realized P&L from exit prices vs entry price."""
        if self.entry_price is None:
            return None
        total = 0.0
        for price, qty in zip(self.exit_prices, self.exit_contracts):
            total += (price - self.entry_price) * qty * 100  # 1 contract = 100 units
        self.realized_pnl = total
        return total
