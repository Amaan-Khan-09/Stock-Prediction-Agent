# Discord Stock Prediction Agent

This agent connects Discord signals to the existing Stock-Prediction-Agent project.

Core workflow:

1. A user posts a signal in the configured Discord signal channel.
2. The agent parses the message as either an equity signal or an options signal.
3. Equity signals run through the project's Stock Price Validation flow.
4. Options signals run through the project's Options Strategy Validation flow.
5. Paper trades are placed only after the matching validation mode approves the signal and Alpaca paper checks pass.
7. Positions bought by this agent are tracked locally.
8. If a tracked position drops 0.5% from the recorded entry price, the agent sends a paper market sell.

Example signals:

```text
buy TSLA qty 2
sell AAPL qty 1
hold MSFT
buy google qty 3
buy dell
buy ford
buy intel
BTO AAPL 240C 08/21 @3.45
Buy SPY 600 CE qty 1
Buy 30 delta AAPL call
```

The parser supports common company-name aliases such as `apple -> AAPL`, `google -> GOOGL`,
`dell -> DELL`, `ford -> F`, and `intel -> INTC`. It also accepts explicit ticker-style
symbols, so users can type direct tickers even when a company-name alias is not listed.

Setup:

```powershell
cd C:\Users\mdama\Stock-Prediction-Agent
venv\Scripts\activate
pip install -r discord_stock_prediction_agent\requirements.txt
notepad discord_stock_prediction_agent\.env
```

For the core setup, create only one Discord text channel:

```text
stock-signals
```

Users send raw signals there. The agent must not post output in this input channel.

Create a separate output channel:

```text
agent-review
```

Prediction reviews, trade confirmations, invalid-input messages, and protection-sell alerts go to `agent-review`.

Fill `discord_stock_prediction_agent\.env` with:

```text
DISCORD_BOT_TOKEN
DISCORD_SIGNAL_CHANNEL_ID
AGENT_REVIEW_CHANNEL_ID
SIGNAL_REVIEW_CHANNEL_ID
ALPACA_API_KEY
ALPACA_SECRET_KEY
POLYGON_API_KEY
```

Options validation:

```text
OPTION_STRIKE_VALIDATION_PROVIDER=polygon_first
```

Exact strike signals such as `AAPL 240C`, `SPY 600 CE`, and `TSLA 290P` are validated with Polygon first. The agent builds the exact Polygon option ticker from the underlying, expiry, call/put side, and strike, then checks historical aggregate bars for that real contract. Delta-style signals continue through the existing delta strategy validation flow.

If Polygon is unavailable or returns no exact-contract bars and the provider is `polygon_first`, the agent can still fall back to the existing Tastytrade validation path. If you want strike signals to depend only on Polygon exact-contract validation, set:

```text
OPTION_STRIKE_VALIDATION_PROVIDER=polygon
```

Optional decision tuning values:

```text
BUY_MIN_RETURN_PCT=0.01
BUY_STRONG_RETURN_PCT=1.0
BUY_EXCELLENT_CONFIDENCE=80
BUY_LOW_RISK=40
BUY_DECISION_SCORE=45
SELL_MIN_RETURN_PCT=-0.01
SELL_STRONG_RETURN_PCT=-1.0
SELL_LOW_CONFIDENCE=50
SELL_HIGH_RISK=60
SELL_DECISION_SCORE=45
```

These values control the equity Stock Price Validation decision gate without changing Python code.

The agent reads `AGENT_REVIEW_CHANNEL_ID` first. These older names also work:

```text
DISCORD_AGENT_REVIEW_CHANNEL_ID=
SIGNAL_REVIEW_CHANNEL_ID=
DISCORD_SIGNAL_REVIEW_CHANNEL_ID=
DISCORD_REVIEW_CHANNEL_ID=
```

Leave `DISCORD_PAPER_LOG_CHANNEL_ID` blank if you want paper trade logs to stay with the normal review output.

Run:

```powershell
venv\Scripts\python.exe -m discord_stock_prediction_agent.discord_agent
```

Safety rules:

- This agent uses Alpaca paper trading endpoints only by default.
- `BUY` defaults to quantity 1 when quantity is not provided.
- `SELL` first checks that Alpaca has shares. If quantity is missing, it sells the full available Alpaca position.
- `BUY` is allowed when the user signal is `BUY` and the AI prediction passes one of these paths: strong positive return, tiny positive return with excellent confidence/risk, or weighted score.
- `SELL` is allowed when the user signal is `SELL` and the AI prediction passes one of these paths: strong negative return, any negative return with low confidence/high risk, or weighted danger score.
- All other cases become `HOLD`, and no paper trade is placed.
- If final output is `SELL` but Alpaca has no shares, no trade is placed.
- The 0.5% protection sell only applies to positions recorded in `agent_state.json` after this agent buys them.
- Each processed signal is stored in `agent_state.json` under `decision_history` so the rules can be tuned from real outcomes over time.
- Closed tracked paper trades are stored under `trade_outcomes`.
- The agent updates `learning_profile` after tracked positions close: losing BUY outcomes make future BUY decisions stricter, while winning outcomes gradually loosen the gate.
- The decision score also uses a benchmark market-regime check. A risk-on benchmark slightly supports BUY decisions; a risk-off benchmark supports SELL/caution.
- News, geopolitics, and world-condition sentiment are not automatic yet. Add a news/sentiment API before using those factors in live scoring.
