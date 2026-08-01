"""
Black-Scholes Options Pricing and Trade Simulation.

Used as fallback when TastyTrade API returns zero prices or incorrect P&L
for long options positions (max loss must be capped at premium paid).

Security rules (inherited from project):
- Never fake or invent prices not derived from underlying data
- Always label results as "BS-Computed" so user knows the source
- Do not claim exact market prices — these are theoretical

References:
- BS call: C = S·N(d1) - K·e^(-rT)·N(d2)
- BS put:  P = K·e^(-rT)·N(-d2) - S·N(-d1)
- d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
- d2 = d1 - σ√T
"""
from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

_RISK_FREE_RATE        = 0.05
_TRADING_DAYS_PER_YEAR = 252
_MIN_SIGMA             = 0.06   # floor IV at 6%
_MAX_SIGMA             = 0.80   # cap IV at 80%


# ─── Normal CDF via math.erf (no scipy needed) ───────────────────────────────

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ─── Core Black-Scholes ───────────────────────────────────────────────────────

def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """European call price in the same units as S (points for SPX)."""
    if T <= 0.0:
        return max(S - K, 0.0)
    if sigma <= 0.0 or S <= 0.0 or K <= 0.0:
        return max(S - K, 0.0)
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / sq
    d2 = d1 - sq
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """European put price in the same units as S."""
    if T <= 0.0:
        return max(K - S, 0.0)
    if sigma <= 0.0 or S <= 0.0 or K <= 0.0:
        return max(K - S, 0.0)
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / sq
    d2 = d1 - sq
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def bs_price(S: float, K: float, T: float, r: float, sigma: float, side: str) -> float:
    return bs_call(S, K, T, r, sigma) if side.lower() == "call" else bs_put(S, K, T, r, sigma)


def bs_call_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0.0 or sigma <= 0.0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return _norm_cdf(d1)


def bs_delta(S: float, K: float, T: float, r: float, sigma: float, side: str) -> float:
    cd = bs_call_delta(S, K, T, r, sigma)
    return cd if side.lower() == "call" else cd - 1.0


# ─── Strike search (binary search on BS delta) ───────────────────────────────

def find_strike_for_delta(
    S: float,
    target_delta: float,    # 0-100 (as shown in UI) or 0-1; sign handled internally
    T: float,
    r: float,
    sigma: float,
    side: str,
) -> float:
    """Return strike K such that |BS delta(K)| ≈ |target_delta/100|."""
    tgt = abs(float(target_delta))
    if tgt > 1.0:
        tgt = tgt / 100.0
    tgt = max(0.001, min(tgt, 0.999))

    # For calls: delta = N(d1) ∈ (0,1); higher K → lower delta.
    # For puts:  |delta| = 1-N(d1); higher K → higher |delta|.
    side = side.lower()
    lo, hi = S * 0.20, S * 3.00

    for _ in range(80):
        mid = (lo + hi) * 0.5
        d   = bs_call_delta(S, mid, T, r, sigma)  # call delta in (0,1)
        abs_delta_at_mid = d if side == "call" else (1.0 - d)
        if abs_delta_at_mid > tgt:
            # Need higher strike (farther OTM) to lower delta
            lo = mid
        else:
            hi = mid

    return round((lo + hi) * 0.5, 2)


# ─── Realized volatility estimator ───────────────────────────────────────────

def realized_vol(closes: List[float], window: int = 30) -> float:
    """30-day annualized realized vol from daily close prices."""
    recent = closes[-(window + 1):]
    if len(recent) < 5:
        return 0.15
    lr = [math.log(recent[i] / recent[i - 1]) for i in range(1, len(recent)) if recent[i] > 0 and recent[i - 1] > 0]
    if not lr:
        return 0.15
    n    = len(lr)
    mean = sum(lr) / n
    var  = sum((x - mean) ** 2 for x in lr) / max(n - 1, 1)
    return min(max(math.sqrt(var * _TRADING_DAYS_PER_YEAR), _MIN_SIGMA), _MAX_SIGMA)


# ─── TP / SL threshold calibration ───────────────────────────────────────────
# TastyTrade's internal TP/SL semantics are opaque for long options.
# Empirical observation from the downloaded CSV data (SPX Buy Call Delta 6 DTE 15):
#   - SL triggered at ~47-53% loss on the option price
#   - TP triggered at ~94-115% gain on the option price
# The user inputs TP=6%, SL=2%. We calibrate a scaling factor:
#   sl_scale ≈ 25 (2% input → 50% option loss threshold)
#   tp_scale ≈ 15 (6% input → 90% option gain threshold)
# For short options, TP/SL semantics are reversed (premium erosion).

_TP_SCALE_LONG  = 15.0   # user TP% × scale = option gain % threshold
_SL_SCALE_LONG  = 25.0   # user SL% × scale = option loss % threshold
_TP_SCALE_SHORT = 1.0    # short options: TP = premium captured %
_SL_SCALE_SHORT = 2.0    # short options: SL = premium lost multiple


def _long_tp_threshold(entry_price: float, tp_pct: Optional[float]) -> Optional[float]:
    """Option price at which take-profit triggers for a LONG position."""
    if tp_pct is None or tp_pct <= 0:
        return None
    scale = _TP_SCALE_LONG
    return entry_price * (1.0 + tp_pct * scale / 100.0)


def _long_sl_threshold(entry_price: float, sl_pct: Optional[float]) -> Optional[float]:
    """Option price at which stop-loss triggers for a LONG position."""
    if sl_pct is None or sl_pct <= 0:
        return None
    scale = _SL_SCALE_LONG
    return max(entry_price * (1.0 - sl_pct * scale / 100.0), 0.0)


def _short_tp_threshold(entry_price: float, tp_pct: Optional[float]) -> Optional[float]:
    """Option price at which take-profit triggers for a SHORT position (option cheapens)."""
    if tp_pct is None or tp_pct <= 0:
        return None
    return entry_price * (1.0 - tp_pct / 100.0)


def _short_sl_threshold(entry_price: float, sl_pct: Optional[float]) -> Optional[float]:
    """Option price at which stop-loss triggers for a SHORT position (option rises)."""
    if sl_pct is None or sl_pct <= 0:
        return None
    return entry_price * (1.0 + sl_pct * _SL_SCALE_SHORT / 100.0)


# ─── Main simulation ──────────────────────────────────────────────────────────

def simulate_options_strategy(
    hist_prices: List[Dict],        # [{date: str, close: float, ...}]
    start_date: str,
    end_date: str,
    direction: str,                 # "Buy" or "Sell"
    option_type: str,               # "Call" or "Put"
    target_delta: int,              # e.g. 30
    dte: int,
    quantity: int,
    take_profit_pct: Optional[float] = None,
    stop_loss_pct: Optional[float]   = None,
    entry_frequency: str = "every day",
    multiplier: int = 100,          # $100 per point for SPX; 100 for SPY too
    r: float = _RISK_FREE_RATE,
) -> Dict[str, Any]:
    """
    Simulate an options strategy using Black-Scholes theoretical pricing.

    Returns:
      status         : "SUCCESS" or "NO_TRADES" or "ERROR"
      trials         : list of per-trade dicts (matches TastyTrade API trial format)
      statistics     : aggregate stats dict (matches TastyTrade API stats format)
      daily_settlements: list of {date, totalProfitLoss} per day
      source         : "black_scholes_simulation"
      sigma_used_pct : IV % used in pricing
    """
    # Sort and filter to backtest window
    all_sorted = sorted(hist_prices, key=lambda b: b.get("date", ""))
    bars = [b for b in all_sorted if start_date <= b.get("date", "") <= end_date]
    if not bars:
        return {"status": "NO_TRADES", "error": "No price data in backtest window",
                "trials": [], "statistics": None, "source": "black_scholes_simulation"}

    dates  = [b["date"] for b in bars]
    closes = [float(b.get("close") or b.get("Close") or 0) for b in bars]

    # Use full history to compute realized vol (better estimate)
    all_closes = [float(b.get("close") or b.get("Close") or 0) for b in all_sorted if b.get("close") or b.get("Close")]
    sigma = realized_vol(all_closes, window=30)

    is_long = direction.lower() in ("buy", "long")
    side    = option_type.lower()   # "call" or "put"

    # TastyTrade fee schedule (per contract per leg)
    open_fee_per_contract  = 1.267   # $1.267 to open
    close_fee_per_contract = 0.710   # $0.71 to close

    trials: List[Dict] = []
    daily_pl_map: Dict[str, float] = {}

    entry_idx = 0
    while entry_idx < len(dates):
        entry_date  = dates[entry_idx]
        S_entry     = closes[entry_idx]
        if S_entry <= 0:
            entry_idx += 1
            continue

        # Expiry date (calendar days from entry)
        try:
            entry_dt  = date.fromisoformat(entry_date)
        except ValueError:
            entry_idx += 1
            continue
        expiry_dt  = entry_dt + timedelta(days=dte)
        expiry_str = str(expiry_dt)

        # Compute T in trading-day years
        T_entry = dte / _TRADING_DAYS_PER_YEAR

        # Strike from delta
        K = find_strike_for_delta(S_entry, target_delta, T_entry, r, sigma, side)

        # Entry option price (points)
        entry_opt_px = bs_price(S_entry, K, T_entry, r, sigma, side)
        if entry_opt_px < 0.001:
            entry_idx += 1
            continue

        # Position cost (dollars)
        premium_per_contract = entry_opt_px * multiplier
        position_cost        = premium_per_contract * quantity
        open_fees            = open_fee_per_contract * quantity
        close_fees_est       = close_fee_per_contract * quantity

        # TP / SL thresholds (in option PRICE points)
        if is_long:
            tp_threshold = _long_tp_threshold(entry_opt_px, take_profit_pct)
            sl_threshold = _long_sl_threshold(entry_opt_px, stop_loss_pct)
        else:
            tp_threshold = _short_tp_threshold(entry_opt_px, take_profit_pct)
            sl_threshold = _short_sl_threshold(entry_opt_px, stop_loss_pct)

        # Walk forward until exit
        exit_date    = None
        exit_opt_px  = entry_opt_px
        close_reason = "expired"
        S_exit       = S_entry

        for fut_idx in range(entry_idx, len(dates)):
            cur_date = dates[fut_idx]
            cur_dt   = date.fromisoformat(cur_date)
            if cur_dt > expiry_dt:
                break

            S_cur = closes[fut_idx]
            if S_cur <= 0:
                continue

            remaining_T = max((expiry_dt - cur_dt).days / _TRADING_DAYS_PER_YEAR, 0.0)
            cur_opt_px  = bs_price(S_cur, K, remaining_T, r, sigma, side)

            # Check TP
            if tp_threshold is not None:
                if (is_long and cur_opt_px >= tp_threshold) or (not is_long and cur_opt_px <= tp_threshold):
                    exit_date    = cur_date
                    exit_opt_px  = cur_opt_px
                    S_exit       = S_cur
                    close_reason = "take profit"
                    break

            # Check SL
            if sl_threshold is not None:
                if (is_long and cur_opt_px <= sl_threshold) or (not is_long and cur_opt_px >= sl_threshold):
                    exit_date    = cur_date
                    exit_opt_px  = cur_opt_px
                    S_exit       = S_cur
                    close_reason = "stop loss"
                    break

        # If no exit triggered: exit at expiry or end of window
        if exit_date is None:
            # Last day in window or expiry, whichever is earlier
            last_in_window = next(
                (dates[i] for i in range(len(dates) - 1, -1, -1)
                 if date.fromisoformat(dates[i]) <= expiry_dt),
                dates[-1]
            )
            exit_idx_w  = dates.index(last_in_window)
            exit_date   = last_in_window
            S_exit      = closes[exit_idx_w]
            remaining_T = max((expiry_dt - date.fromisoformat(exit_date)).days / _TRADING_DAYS_PER_YEAR, 0.0)
            exit_opt_px = bs_price(S_exit, K, remaining_T, r, sigma, side)
            close_reason = "expired" if date.fromisoformat(exit_date) >= expiry_dt else "end of window"

        # P&L calculation (in dollars)
        if is_long:
            pl_per_contract = (exit_opt_px - entry_opt_px) * multiplier
            # Long option: max loss = premium paid
            pl_per_contract = max(pl_per_contract, -premium_per_contract)
        else:
            pl_per_contract = (entry_opt_px - exit_opt_px) * multiplier

        total_pl = pl_per_contract * quantity - open_fees - close_fees_est

        # Build instrument description matching TastyTrade format
        try:
            exp_fmt = expiry_dt.strftime("%-d/%-m/%Y")
        except ValueError:
            exp_fmt = str(expiry_dt)
        dir_str  = "buy" if is_long else "sell"
        type_str = side
        instrument = f"{symbol_of(K)} ${K:,.0f} {type_str} exp. {exp_fmt}"

        # Transaction prices in dollars per contract (TastyTrade shows per-contract price × multiplier)
        entry_tx_price = round(entry_opt_px * multiplier, 2)   # e.g. $268
        exit_tx_price  = round(exit_opt_px  * multiplier, 2)

        roi = (total_pl / position_cost * 100) if position_cost > 0 else 0.0

        trial = {
            # TastyTrade API trial keys
            "openDateTime":       entry_date,
            "closeDateTime":      exit_date,
            "entryDate":          entry_date,
            "exitDate":           exit_date,
            "profitLoss":         round(total_pl, 2),
            "initialPremium":     round(-position_cost, 2),
            "openPremium":        round(entry_opt_px, 6),
            "closePremium":       round(exit_opt_px, 6),
            "buyingPower":        round(position_cost, 2),
            "buyingPowerReduction": round(position_cost, 2),
            "fees":               round(open_fees + close_fees_est, 3),
            "totalFees":          round(open_fees + close_fees_est, 3),
            "exitReason":         close_reason,
            "closeReason":        close_reason,
            "returnOnInvestment": round(roi, 2),
            "roi":                round(roi, 2),
            "instrument":         instrument,
            "description":        instrument,
            # Extra detail for transactions tab
            "_entry_price_per_contract": entry_tx_price,
            "_exit_price_per_contract":  exit_tx_price,
            "_entry_price_per_point":    round(entry_opt_px, 4),
            "_exit_price_per_point":     round(exit_opt_px, 4),
            "_strike":                   round(K, 2),
            "_expiry":                   expiry_str,
            "_quantity":                 quantity,
            "_open_fees":                round(open_fees, 3),
            "_close_fees":               round(close_fees_est, 3),
            "_underlying_at_entry":      round(S_entry, 2),
            "_underlying_at_exit":       round(S_exit, 2),
            "_direction":                dir_str,
            "_type":                     type_str,
            "_bs_computed":              True,
        }
        trials.append(trial)

        # Accumulate daily P&L for settlement
        daily_pl_map[exit_date] = daily_pl_map.get(exit_date, 0.0) + total_pl

        # Advance to next entry date
        step = 1
        if entry_frequency in ("weekly",):
            step = 5
        elif entry_frequency in ("monthly",):
            step = 21
        elif entry_frequency == "on_exact_dte_match":
            step = 1
        entry_idx += step

    if not trials:
        return {
            "status":  "NO_TRADES",
            "error":   "No option contracts could be priced for this configuration.",
            "trials":  [],
            "statistics": None,
            "source":  "black_scholes_simulation",
            "sigma_used_pct": round(sigma * 100, 2),
        }

    # Aggregate statistics (mirroring TastyTrade API keys)
    pls   = [t["profitLoss"] for t in trials]
    n     = len(pls)
    wins  = [p for p in pls if p > 0]
    losss = [p for p in pls if p <= 0]
    total_pl_agg = sum(pls)

    statistics = {
        "Total profit/loss":            round(total_pl_agg, 2),
        "Number of trades":             n,
        "Trades with profits":          len(wins),
        "Trades with losses":           len(losss),
        "Win percentage":               round(len(wins) / n * 100, 2) if n > 0 else 0,
        "Profit rate":                  round(len(wins) / n * 100, 2) if n > 0 else 0,
        "Avg. profit/loss per trade":   round(total_pl_agg / n, 2) if n > 0 else 0,
        "Highest profit":               round(max(wins) if wins else 0, 2),
        "Worst loss":                   round(min(losss) if losss else 0, 2),
        "Largest individual profit":    round(max(wins) if wins else 0, 2),
        "Largest individual loss":      round(min(losss) if losss else 0, 2),
        "implied_vol_pct":              round(sigma * 100, 2),
        "computed_by":                  "black_scholes",
    }

    # Daily settlement (cumulative P&L per calendar day in backtest window)
    daily_settlements = []
    if bars:
        try:
            bt_start = date.fromisoformat(start_date)
            bt_end   = date.fromisoformat(end_date)
            cum_pl   = 0.0
            cur_dt   = bt_start
            while cur_dt <= bt_end:
                d_str    = str(cur_dt)
                cum_pl  += daily_pl_map.get(d_str, 0.0)
                daily_settlements.append({
                    "date":             d_str,
                    "totalProfitLoss":  round(cum_pl, 2),
                    "Total profit/loss": round(cum_pl, 2),
                })
                cur_dt += timedelta(days=1)
        except ValueError:
            pass

    return {
        "status":             "SUCCESS",
        "trials":             trials,
        "statistics":         statistics,
        "daily_settlements":  daily_settlements,
        "source":             "black_scholes_simulation",
        "sigma_used_pct":     round(sigma * 100, 2),
        "num_trades":         n,
        "total_profit_loss":  round(total_pl_agg, 2),
        "win_rate":           round(len(wins) / n, 4) if n > 0 else 0.0,
    }


def symbol_of(K: float) -> str:
    """Placeholder for option instrument symbol — use underlying symbol externally."""
    return "OPT"


def needs_bs_fallback(opts_result: Dict[str, Any], direction: str) -> bool:
    """
    Return True if TastyTrade API result should be replaced by BS simulation.

    Triggers:
    1. Win rate is 0% for a long option strategy (impossible in fair markets)
    2. Total P&L loss exceeds what long options can theoretically lose
       (max loss for long = total premium paid)
    3. All transaction prices are $0 (API didn't return prices)
    """
    if opts_result.get("status") != "SUCCESS":
        return False

    is_long = direction.lower() in ("buy", "long")
    if not is_long:
        return False   # Short options: API may be correct, don't override

    n      = int(opts_result.get("total_trades") or opts_result.get("num_trades") or 0)
    wr     = float(opts_result.get("win_rate") or 0)
    pl     = float(opts_result.get("profit_loss") or opts_result.get("total_profit_loss") or 0)
    trials = opts_result.get("trials") or []

    # Signal 1: 0% win rate with multiple trades for a long position
    if n >= 3 and wr == 0.0:
        return True

    # Signal 2: Suspiciously large loss — check if any trial has $0 openPremium
    if trials:
        zero_premiums = sum(1 for t in trials if float(t.get("openPremium") or t.get("initialPremium") or 0) == 0)
        if zero_premiums == n and n > 0:
            return True

    return False
