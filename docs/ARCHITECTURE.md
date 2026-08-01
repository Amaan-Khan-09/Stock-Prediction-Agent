# Architecture — Flex Data / Ajay AI Finance Analyst

> This project is called "Flex Data / Ajay AI Finance Analyst". Not Xeltrix.

## Overview

A multi-engine, deterministic-first financial intelligence system with tastytrade options backtesting, a live stock analysis AI, portfolio analytics, and leakage-safe equity backtesting.

## Entry Point

```
streamlit run streamlit_app.py
```

## Directory Structure

```
AI FINANCIAL ANALYST SYSTEM/
├── streamlit_app.py          ← Streamlit UI entry point (all 8 pages)
├── stock_analysis_agent.py   ← Core scoring + live analysis engine
├── evaluation_engine.py      ← Historical backtesting + portfolio intelligence
├── portfolio_manager.py      ← Portfolio facade (3-agent pipeline)
├── portfolio_orchestrator.py ← Per-stock analysis loop
├── portfolio_recommendation_agent.py  ← Gemini-powered recommendations
├── portfolio_parser.py       ← Portfolio text parser
├── recommendation_accuracy_engine.py  ← Ground truth evaluation
├── user_portfolio.py         ← Ajay's real portfolio (standalone script)
│
├── src/
│   ├── config/
│   │   └── settings.py       ← Centralised .env config (Settings dataclass)
│   ├── models/
│   │   ├── backtest_models.py ← BacktestLeg, BacktestPayload, BacktestResult, etc.
│   │   ├── portfolio_models.py← Holding, PortfolioMetrics, PortfolioAnalysisResult
│   │   └── stock_models.py   ← IntelligenceScore, StockVerdict, PredictionRecord
│   ├── services/
│   │   ├── tastytrade_auth_service.py     ← OAuth + token refresh
│   │   ├── tastytrade_backtester_service.py← Options backtest (equity-option)
│   │   ├── internal_backtester_service.py ← Equity backtest (buy/hold, MA)
│   │   ├── portfolio_service.py           ← Concentration + correlation
│   │   ├── prediction_validator_service.py← Direction accuracy evaluation
│   │   ├── ai_report_service.py           ← Gemini narrative + fallback
│   │   └── market_data_service.py         ← RapidAPI TradingView wrapper
│   ├── ui/
│   │   └── streamlit_components.py        ← Reusable UI components
│   ├── utils/
│   │   ├── logging_utils.py  ← Structured logging (no token leakage)
│   │   ├── validation_utils.py← Input validation helpers
│   │   ├── date_utils.py     ← Date formatting helpers
│   │   └── math_utils.py     ← Finance math with Decimal
│   └── tests/
│       ├── test_tastytrade_payload.py  ← 7 payload tests
│       ├── test_tastytrade_parser.py   ← 6 parser/validation tests
│       ├── test_internal_backtester.py ← 4 equity backtest tests
│       └── test_portfolio_service.py   ← 4 portfolio tests
│
├── tools/
│   ├── __init__.py
│   ├── tradingview_price.py
│   ├── tradingview_market_data.py
│   ├── tradingview_technical_analysis.py
│   ├── tradingview_news.py
│   └── ...
│
├── data/
│   └── stock_prices_daily.csv  ← OHLCV dataset (Date, Ticker, Open, High, Low, Close, Volume)
│
├── docs/
│   ├── TASTYTRADE_INTEGRATION.md
│   └── ARCHITECTURE.md
│
├── .env                  ← Secrets (never commit)
├── .env.example          ← Template (safe to commit)
├── .gitignore
└── requirements.txt
```

## 8 UI Pages

| Page | Function | Description |
|------|----------|-------------|
| Ajay Demo | `institutional_backtesting_interface()` | Institutional backtest + accuracy |
| Stock Research | `stock_analysis_interface()` | Live stock analysis |
| Portfolio Analysis | `portfolio_analysis_interface()` | Portfolio intelligence |
| Tastytrade Options Backtesting | `tastytrade_options_backtesting_interface()` | Real options backtest |
| Internal Backtesting | `internal_backtesting_interface()` | Equity backtest from CSV |
| Prediction Validation | `prediction_validation_interface()` | Direction accuracy |
| AI Report | `ai_report_interface()` | Gemini narrative |
| Evaluation Lab | `evaluation_lab_interface()` | Full backtesting suite |

## Intelligence Engines (Deterministic)

All BUY/HOLD/SELL decisions are **always deterministic**, never from LLM output.

| Engine | Weight | Source |
|--------|--------|--------|
| Fundamental | 20% | RapidAPI (P/E, EPS, revenue) |
| Technical | 25% | RapidAPI indicators |
| Valuation | 15% | RapidAPI (P/B, EV/EBITDA) |
| Macro | 10% | RapidAPI (GDP, rates) |
| Sentiment | 10% | RapidAPI (news, community) |
| Risk penalty | −20% | Volatility, drawdown |

Verdict thresholds: `score ≥ 67 → BUY`, `score ≤ 33 → SELL`, else `HOLD`.

## Data Flow

```
User Input
    ↓
streamlit_app.py
    ↓
StockAnalysisAgent.analyze_stock()
    ↓
compute_intelligence_scores()         ← RapidAPI data
    ↓
Deterministic scoring (6 engines)
    ↓
Composite score → Verdict (BUY/HOLD/SELL)
    ↓
Gemini (narrative only, no override)
    ↓
render_stock_intelligence()
```

## Tastytrade Data Flow

```
User Input (symbol, dates, DTE, delta)
    ↓
tastytrade_options_backtesting_interface()
    ↓
run_options_backtest()
    ↓
build_equity_option_short_put_payload()  ← type="equity-option" REQUIRED
    ↓
create_backtest() → POST /backtests
    ↓
poll_backtest() → GET /backtests/{id}
    ↓
parse_backtest_result()
    ↓
validate_backtest_success()  ← STRICT gate
    ↓
extract_backtest_summary() → UI render
```

## Security Rules

- Never hardcode API keys, tokens, or secrets anywhere in source code.
- Use `.env` variables only — loaded via `python-dotenv`.
- Never log Authorization headers or token values.
- Never expose tokens in Streamlit error messages.
- Never place live-money trades. The Discord extension implements Alpaca paper-order endpoints and rejects non-paper broker configuration.
- Do not use `https://api.cert.tastyworks.com` for this integration.
- Do not commit `.env` — it is in `.gitignore`.
