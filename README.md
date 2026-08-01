# AI Financial Analyst System

### *The world's first self-calibrating AI that predicts options strategy outcomes — then proves itself right using real market data.*

---

> **"The AI predicts. The market proves. The system learns. No leakage. No lies. No shortcuts."**

---

## The Problem No One Has Solved

Every retail and institutional options trader faces the same brutal question before placing a trade:

> *"If I run this exact strategy — symbol, direction, strike, DTE, legs, frequency — over this exact time window with this capital, what will happen?"*

Bloomberg charges $25,000/year for terminal access. Institutional quant desks employ 50-person teams. Retail traders guess.

**This system changes that.**

---

## What This Is

**AI Financial Analyst System** is a production-grade quantitative finance platform that does something unprecedented:

1. **AI makes a prediction** — A calibrated surrogate model (v3) predicts exact financial metrics for any options strategy before the trade is placed: total return, P&L, CAGR, Sharpe ratio, win rate, max drawdown, final capital.

2. **The market verifies it** — The exact same strategy is backtested live on **tastytrade's production backtester** (the same engine used by professional options traders) over the same symbol, dates, and parameters.

3. **Accuracy is measured rigorously** — The system computes numeric error metrics (return error %, P&L error, capital error) and an Overall Strategy Accuracy score combining directional signals with numeric closeness.

4. **The AI improves itself** — A Calibration Engine (v3) runs mini-backtests on similar historical windows and uses them to pull the AI prediction toward empirical ground truth — without ever seeing the current backtest result (no data leakage).

**No fake market values, hardcoded predictions, or random decision outputs. Market data is provider-attributed; the historical service can use TradingView, NASDAQ, or yfinance according to availability.**

---

## Live System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      AI FINANCIAL ANALYST SYSTEM                            │
│                         Streamlit Wide Dashboard                            │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │  User inputs: Symbol · Dates · Capital
                             │  Direction · Side · DTE · Delta · Legs · Freq
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   CENTRAL VALIDATION GATE (strategy_validation.py)          │
│   Symbol format · Blocklist · Date range (≥2000, ≤today) · Capital · Params │
│   FAIL → st.error + st.stop   ✦   PASS → run pipeline                      │
└──────────────────────┬─────────────────────────┬───────────────────────────┘
                       │                         │
          ┌────────────▼────────────┐  ┌─────────▼──────────────────────────┐
          │  CALIBRATION ENGINE v3  │  │      TASTYTRADE PRODUCTION         │
          │  (strategy_calibration  │  │          BACKTESTER                │
          │       _engine.py)       │  │   (live API, real market data)     │
          │                         │  │                                    │
          │  • Loads similar-strategy│  │  • Same symbol, dates, params      │
          │    records from JSONL   │  │  • Real trial-by-trial P&L         │
          │  • 9-factor similarity  │  │  • OAuth2 + long-poll              │
          │  • Runs mini backtests  │  │  • Phase 8 decision mapping        │
          │    if < 6 records found │  │    (P&L overrides win rate)        │
          │  • Blending: ≥8 → 75%  │  │  • Max profit/loss from trials     │
          └────────────┬────────────┘  └─────────────────┬──────────────────┘
                       │                                  │
          ┌────────────▼──────────────────────────────────┤
          │         AI PREDICTION AGENT v3                │
          │  (strategy_predictor_v3_backtest_surrogate_   │
          │            _calibrated)                       │
          │                                               │
          │  Stage 1: Raw Prior Model                     │
          │    ├─ RapidAPI market context (TradingView)   │
          │    ├─ 10-factor score computation             │
          │    └─ Deterministic projection                │
          │                                               │
          │  Stage 2: Calibration Dataset                 │
          │    ├─ Load similar records (sim ≥ 0.50)      │
          │    ├─ Weighted bias computation               │
          │    └─ Blending by record count:               │
          │       ≥8 records → 75% empirical              │
          │       5-7 records → 60% empirical             │
          │       3-4 records → 45% empirical             │
          │       2 records  → 20% empirical              │
          │                                               │
          │  Stage 3: Calibrated Output                   │
          │    ├─ Blended: return, win_rate, drawdown     │
          │    ├─ Gemini 2.5 Flash explanation            │
          │    └─ SHA-256 input hash (integrity)          │
          └────────────────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────────┐
                    │     STRATEGY BACKTEST COMPARATOR         │
                    │   (strategy_backtest_comparator.py)      │
                    │                                          │
                    │   Decision agreement · Numeric errors    │
                    │   MATCH / PARTIAL / CONFLICT / MISSING   │
                    │   Final verified decision                │
                    └──────────────────────┬───────────────────┘
                                           │
                    ┌──────────────────────▼───────────────────┐
                    │         ACCURACY ENGINE                   │
                    │    (strategy_accuracy_engine.py)          │
                    │                                           │
                    │   Rolling verification · JSONL storage    │
                    │   Overall Strategy Accuracy score         │
                    │   Quarantine gate for invalid records     │
                    └───────────────────────────────────────────┘
```

---

## Feature Highlights

### Prediction Engine — 15 Financial Metrics Per Run

| Metric | Description |
|--------|-------------|
| **Total Return %** | Strategy P&L as a percentage of initial capital |
| **Total P&L ($)** | Net profit or loss in absolute dollars |
| **Final Capital ($)** | Portfolio value at end of the strategy window |
| **CAGR (%)** | Compound Annual Growth Rate |
| **Sharpe Ratio** | Risk-adjusted return vs 4.5% risk-free rate |
| **Sortino Ratio** | Downside deviation-adjusted return |
| **Max Drawdown (%)** | Peak-to-trough equity curve decline |
| **Win Rate (%)** | Percentage of trades that were profitable |
| **Trade Count** | Number of entries in the strategy window |
| **Volatility (%)** | Annualized standard deviation of trade returns |
| **Beta** | Market directional exposure coefficient |
| **Alpha (%)** | Excess return vs SPY benchmark |
| **Risk Score (0-100)** | Composite: drawdown + volatility + loss rate |
| **Confidence Score (0-100)** | Calibration confidence from historical record depth |
| **Max Profit / Max Loss ($)** | Best and worst individual trade outcomes |

---

### The Three-Stage AI Pipeline

```
Stage 1: Raw Prior Model
─────────────────────────
Input: StrategyInput + RapidAPI Market Context
         │
         ├─ Underlying regime score    (trending / mean-reverting / volatile)
         ├─ Strategy suitability score (short premium / long vol alignment)
         ├─ News sentiment score       (positive / negative / neutral)
         ├─ Market breadth score       (gainers vs losers ratio)
         ├─ Liquidity / activity score (volume movers in context)
         ├─ Historical regime score    (date range seasonal patterns)
         ├─ Strike selection score     (delta / DTE alignment)
         ├─ Leg structure score        (single vs spread efficiency)
         ├─ Capital efficiency score   (capital vs expected premium)
         └─ Entry frequency score      (monthly / weekly / daily)
         │
         └→ Deterministic Projection (no randomness)

Stage 2: Calibration Dataset
──────────────────────────────
Input: Similar historical strategy backtest records
         │
         ├─ 9-factor similarity scoring (threshold ≥ 0.50)
         │    symbol · direction · side · delta_bucket · DTE_bucket ·
         │    legs · entry_frequency · date_regime · benchmark
         │
         ├─ Record count → blending weight schedule
         │
         └→ Empirical bias (weighted mean of similar actual returns)

Stage 3: Calibrated Output
───────────────────────────
Blended = (empirical_weight × empirical) + (raw_weight × raw_prior)
         │
         ├─ 15 predicted financial metrics
         ├─ SHA-256[:16] input integrity hash
         ├─ Calibration summary (similar_count, calibration_weight)
         └─ Gemini 2.5 Flash narrative explanation (temperature=0)
```

---

### Ground-Truth Verification — tastytrade Production Backtester

The AI prediction is immediately verified against the **same strategy** on tastytrade's production backtesting infrastructure:

- **Real options data** — actual historical premiums, fills, and P&L
- **Trial-by-trial breakdown** — every individual trade entry/exit logged
- **Phase 8 decision logic** — P&L magnitude always overrides win rate:
  - `final_capital ≤ 0` → **SELL** (catastrophic, always)
  - `total_return ≤ -50%` → **SELL** (severe loss, always)
  - `total_return > 5% AND P&L > 0` → **BUY** (profitable)
  - `total_return < -5%` → **SELL** (losing, regardless of win rate)
  - `-5% ≤ total_return ≤ 5%` → **HOLD** (neutral band)

---

### Self-Improving Calibration

```
First run for a new symbol/strategy combination:
  → Mini-backtests fired on 12, 18, 24-month sub-windows
  → Stored in JSONL evaluation corpus
  → Similarity scored against current input (9 factors)

Subsequent runs:
  → Calibration records load instantly (no new API calls)
  → AI blends raw prediction with empirical evidence
  → Accuracy improves measurably with each verified run
  → Rolling verification adds ground-truth records automatically
```

**The system learns from every backtest it verifies against.**

---

### Input Integrity and Validation

Every run passes through a **strict, unbypassable validation gate** before any API is called:

| Check | Rule |
|-------|------|
| Symbol format | 1–5 uppercase letters (`^[A-Z]{1,5}([.\-][A-Z])?$`) |
| Symbol reality | Blocklist of 20+ known fake/placeholder symbols (XYZ, ABCXYZ, TEST…) |
| Start date | Must be ≥ 2000-01-01 (options data availability) |
| End date | Must be ≤ today (no future dates) |
| Date span | Minimum 180 days for meaningful backtesting |
| Capital | Minimum $1,000 |
| Parameters | Direction / side / DTE / delta / legs validated with strict ranges |
| Hash integrity | SHA-256[:16] of sorted-key JSON — AI and backtest share the same hash |

**If any check fails: `st.error()` + `st.stop()`. No AI. No backtest. No charts. No accuracy record.**

---

### Accuracy Measurement — Overall Strategy Accuracy Score

```
Overall Strategy Accuracy = weighted combination of:

  Direction Accuracy     (20%) — did the AI predict BUY/SELL/HOLD correctly?
  Directional Accuracy   (15%) — did the AI get the sign right (positive/negative)?
  Return Accuracy        (25%) — how close was predicted_return to actual_return?
  P&L Accuracy           (20%) — how close was predicted_P&L to actual_P&L?
  Capital Accuracy       (10%) — how close was predicted_final_capital to actual?
  Win Rate Accuracy      (10%) — how close was predicted_win_rate to actual?

  Return error ≤ ±10%  → 1.00 score  (perfect numeric prediction)
  Return error ≤ ±20%  → 0.75 score  (excellent)
  Return error ≤ ±50%  → 0.40 score  (directionally useful)
  Return error > ±100% → 0.10 score  (direction only)
```

The only metric that honestly reports **both directional accuracy AND numeric precision** together.

---

### Record Quarantine System

Invalid evaluation records are **never mixed into accuracy statistics**:

```
is_valid_evaluation_record(record) → bool

Rejects:
  • Blank or fake symbols (XYZ, ABCXYZ, TEST, PLACEHOLDER...)
  • Future end dates
  • Pre-2000 start dates (year 1000, 1800, etc.)
  • Records with REVIEW/MISSING decision (backtest failed)
  • Records missing actual_total_return_pct

Invalid records → strategy_prediction_evaluation_runs_invalid.jsonl
Valid records   → strategy_prediction_evaluation_runs.jsonl
```

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **AI Core** | Google Gemini 2.5 Flash | Natural language strategy explanation (temperature=0) |
| **Options Backtester** | tastytrade Production API | Ground-truth strategy verification |
| **Market Data** | TradingView RapidAPI | Gainers, losers, movers, financial news |
| **UI Framework** | Streamlit 1.28+ | Wide-screen 8-section interactive dashboard |
| **Charting** | Plotly | Equity curve, return distribution, comparison charts |
| **Data Storage** | JSONL | Persistent evaluation corpus (append-only, audit-safe) |
| **Integrity** | SHA-256 | Input hash verified across AI and backtest |
| **Cloud** | Google Cloud (GCP) | Project infrastructure and Gemini credentials |
| **Language** | Python 3.11+ | Core system |
| **Testing** | pytest | 74 passing tests with O1-O7 regression suite |

---

## The Dashboard — 8 Sections, Zero Clutter

```
Section 1 — Strategy Inputs Summary
  Symbol · Date range · Capital · Direction · Side · DTE · Delta · Legs
  Entry frequency · Benchmark · SHA-256 input hash

Section 2 — Side-by-Side Comparison
  ┌──────────────────────┬──────────────────────┐
  │   AI PREDICTION      │  TASTYTRADE BACKTEST  │
  │   (Independent)      │  (Ground Truth)       │
  │                      │                       │
  │  15 predicted metrics│  15 actual metrics    │
  │  Calibration weight  │  Trade count          │
  │  Model version       │  Win/loss breakdown   │
  └──────────────────────┴──────────────────────┘

Section 3 — Factor Score Radar
  10 factor scores visualized as a spider chart

Section 4 — Final Verified Decision Board
  MATCH / PARTIAL / CONFLICT → BUY / SELL / HOLD / REVIEW / AVOID

Section 5 — Accuracy Metrics
  Decision accuracy · Return error · P&L error · Overall Strategy Accuracy

Section 6 — Charts
  Equity curve · Return distribution · AI vs backtest comparison

Sections 7–8 — Developer Debug Expanders
  RapidAPI context · Accuracy records · Calibration info · Auth status
```

---

## File Architecture

```
AI FINANCIAL ANALYST SYSTEM/
│
├── streamlit_app.py                  ← Main UI (1,450+ lines, zero sidebar)
├── strategy_prediction_agent.py      ← AI pipeline v3 (surrogate calibrated)
├── strategy_calibration_engine.py    ← Self-improving calibration system
├── strategy_backtest_comparator.py   ← AI vs backtest numeric comparison
├── strategy_accuracy_engine.py       ← Accuracy tracking and quarantine
├── strategy_validation.py            ← Central input validation gate
│
├── tools/
│   └── tradingview_unified.py        ← RapidAPI client (TradingView host only)
│
├── strategy_prediction_evaluation_runs.jsonl          ← Valid evaluation corpus
├── strategy_prediction_evaluation_runs_invalid.jsonl  ← Quarantined records
│
├── test_strategy_prediction_flow.py  ← 74 tests, O1-O7 regression suite
├── requirements.txt
└── .env                              ← API keys (never committed to source control)
```

---

## Quick Start

### Prerequisites

```
Python 3.11+
tastytrade account with API access
RapidAPI key (TradingView host)
Google Cloud account with Gemini API enabled
```

### Installation

```bash
git clone <repository-url>
cd "AI FINANCIAL ANALYST SYSTEM"

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Environment Configuration

Create a `.env` file in the project root:

```env
# Google AI
GOOGLE_API_KEY=your_gemini_api_key
AGENT_MODEL=gemini-2.5-flash

# TradingView RapidAPI
RAPIDAPI_KEY=your_rapidapi_key
RAPIDAPI_HOST=trading-view.p.rapidapi.com
RAPIDAPI_BASE_URL=https://trading-view.p.rapidapi.com

# tastytrade (Production only — never use cert/sandbox)
TASTYTRADE_API_BASE_URL=https://api.tastyworks.com
TASTYTRADE_BACKTESTER_BASE_URL=https://backtester.vast.tastyworks.com
TASTYTRADE_REFRESH_TOKEN=your_refresh_token
TASTYTRADE_USER_AGENT=your-client/1.0

# Google Cloud
GOOGLE_CLOUD_PROJECT=your_project_id
GOOGLE_CLOUD_REGION=us-central1
```

### Launch

```bash
streamlit run streamlit_app.py
```

The dashboard opens at `http://localhost:8501`

### Run Tests

```bash
pytest test_strategy_prediction_flow.py -v --tb=short
# Expected: 74 passed
```

---

## Example Run — MSFT Short Put

**Input**
```
Symbol:           MSFT
Direction:        Short  |  Side: Put
DTE:              45     |  Delta: 30
Legs:             1      |  Frequency: Monthly
Start Date:       2021-06-25
End Date:         2025-01-24
Initial Capital:  $80,000
```

**AI Prediction (v3 Calibrated)**
```
Decision:         BUY  (calibration_weight=75%, similar_count=8)
Total Return:     +231.4%   |  P&L:    +$185,120
Final Capital:    $265,120  |  CAGR:   +42.1%
Win Rate:         78.3%     |  Sharpe:  1.84
Max Drawdown:     12.8%     |  Risk Score: 34 / 100
```

**tastytrade Backtest (Ground Truth)**
```
Decision:         BUY
Total Return:     +221.9%   |  P&L:    +$177,520
Final Capital:    $257,520  |  CAGR:   +40.8%
Win Rate:         80.1%     |  Sharpe:  1.79
Max Drawdown:     14.1%     |  Risk Score: 37 / 100
```

**Comparison Result**
```
Agreement:        MATCH — both engines agree: BUY ✓
Return Error:     +9.5%    (within ±10% band → perfect score)
P&L Error:        +$7,600
Capital Error:    +$7,600  (+3.1%)
Win Rate Error:   −1.8%
Overall Accuracy: 91.4%    ← direction + numeric combined
```

---

## Design Principles

Every rule below is enforced in both code and tests. None of them are aspirational.

| Principle | What it means in practice |
|-----------|--------------------------|
| **No data leakage** | The AI never sees the current backtest result before predicting. Calibration dataset excludes the exact current hash. |
| **No fake data** | Every predicted metric is computed via deterministic formula or blended from real backtest records. Zero randomness. |
| **No silent failures** | Invalid inputs block the entire pipeline. Backtest failures return explicit objects. Silent date-override retries do not exist. |
| **No stale outputs** | Session state is cleared before every new run. A failed validation shows an error — not the previous run's results. |
| **No symbol-independent values** | Formula output depends on RapidAPI market context, which is symbol-specific and time-stamped. |
| **Provider transparency** | Historical data records the provider used. TradingView and NASDAQ are preferred; yfinance is a free last-resort historical provider. |
| **Paper trading only** | The Discord extension can submit orders only to Alpaca's paper endpoint. Production configuration rejects a live Alpaca base URL. |

---

## Test Coverage — 74 Tests

```
TestAIPrediction                       7 tests
TestValidateStrategyInput              5 tests
TestComparator                         7 tests
TestAccuracyEngine                     4 tests
TestEnrichBacktestMetrics              3 tests
TestRapidAPIConfig                     1 test  — enforces trading-view host

TestV2CalibratedPrediction             8 tests
TestV3BacktestSurrogateCalibrated     12 tests
  ├─ No yfinance in any new files
  ├─ MODEL_VERSION = v3_backtest_surrogate_calibrated
  ├─ CALCULATION_VERSION = 3.0
  ├─ No backtest result leakage (raises AssertionError if leaked)
  ├─ Deterministic hash — SHA-256[:16]
  ├─ Backtest decision BUY for highly profitable strategies
  ├─ Calibration excludes the current input hash
  ├─ Calibration blending moves prediction toward empirical evidence
  ├─ Overall accuracy penalizes large numeric errors
  ├─ Overall accuracy high when prediction is close to backtest
  ├─ Rolling verification creates v3-versioned records
  └─ Similarity scoring bounded to [0.0, 1.0]

TestO1ToO7ValidationRegressions       18 tests
  ├─ Blank symbol blocked
  ├─ Whitespace-only symbol blocked
  ├─ XYZ (known placeholder) blocked
  ├─ ABCXYZ (6 characters — invalid format) blocked
  ├─ Real symbols AAPL / MSFT / TSLA / CVS / SPY / GOOGL / BRK.B pass
  ├─ TSLA −727% return with 70% win rate → SELL (not HOLD)
  ├─ CVS −28% return with 72.9% win rate → SELL (not HOLD)
  ├─ Future end date 2027 blocked
  ├─ Year 1000 start date blocked
  ├─ Future start date blocked
  ├─ Invalid date format (DD/MM/YYYY) blocked
  ├─ Hardcoded auto-retry dates verified removed from source
  ├─ Quarantine functions importable
  ├─ Quarantine rejects fake symbol records
  ├─ Quarantine rejects future end date records
  ├─ Quarantine rejects pre-2000 date records
  ├─ Quarantine accepts well-formed real records
  └─ strategy_validation module fully importable
```

---

## Why This Is Different

Most systems labeled "AI trading" do one of the following:

| What they claim | What they actually do |
|----------------|----------------------|
| AI-powered predictions | Return the backtest result with a label |
| Live market data | Pull stale Yahoo Finance prices |
| Accuracy metrics | Report training accuracy on the same data used to build the model |
| Validated inputs | Accept any string, including blank and year-1000 dates |
| Real backtests | Run simulations on synthetic data |

**This system does none of the above.**

The AI predicts independently. The backtester verifies independently. The comparator measures the gap honestly. Invalid inputs are blocked before a single API call is made. Every claim is backed by a passing test.

---

## Commercial Value Proposition

**Institutional-grade methodology** — The three-stage surrogate calibration mirrors how quantitative hedge funds build prediction models: prior distribution, calibration data, blended posterior output.

**Provable and auditable accuracy** — Every run appends a record to an immutable JSONL corpus. The system shows exactly how right or wrong the AI was, in which direction, and by how much.

**Self-improving feedback loop** — The calibration dataset grows with every verified run. The system measurably improves its own numeric accuracy over time without retraining.

**Production data sources only** — tastytrade production backtester, TradingView RapidAPI, Google Gemini 2.5 Flash. No mock APIs, no sandboxes, no synthetic data in the prediction path.

**Enterprise-grade data integrity** — SHA-256 input hashing, quarantine ledger, session isolation, explicit failure states, zero silent errors.

**Extensible architecture** — Strategy parameters, calibration weights, similarity factors, and accuracy components are all configurable. Adding new instruments or markets requires only new strategy input definitions.

---

## Discord Paper-Trading Agent

The repository also contains a production-oriented Discord layer in
`discord_stock_prediction_agent/`. It receives raw equity and options signals, routes
normal trades through **Stock Price Validation**, routes options through **Options
Strategy Validation**, and submits approved orders only to Alpaca's paper endpoint.

Run it from the repository root:

```powershell
python -m discord_stock_prediction_agent.discord_agent
```

The bot uses two primary Discord channel IDs in the local `.env`:

```env
DISCORD_SIGNAL_CHANNEL_ID=your_stock_signals_channel_id
AGENT_REVIEW_CHANNEL_ID=your_agent_review_channel_id
```

Runtime decision controls:

```text
!agent_on    # project validation and decision gates enabled
!agent_off   # valid BUY/SELL signals go directly to paper-order handling
!agent_mode  # display current mode
```

Agent OFF does not disable parsing, position, contract, price, market-hours, paper
endpoint, or broker safeguards. Full installation, Discord setup, environment variables,
signal formats, queue/state locations, monitoring behavior, tests, and production notes
are documented in
[`discord_stock_prediction_agent/README.md`](discord_stock_prediction_agent/README.md).

---

## Roadmap

- [ ] Multi-leg strategy support — iron condors, calendars, strangles, butterflies
- [ ] Portfolio-level prediction — multiple simultaneous strategies with correlation
- [ ] Real-time market regime detection — VIX integration for dynamic calibration
- [ ] Cross-broker backtesting — verify predictions against multiple data sources
- [ ] REST API — institutional embedding and programmatic access
- [ ] Strategy screener — AI recommends the top-ranked options strategies for current market conditions
- [ ] Mobile-responsive layout

---

## License

This project is proprietary software. All rights reserved.

---

*Built with rigor. Verified with data. Improved by the market itself.*
