# Rich stock order fields

The Discord agent deterministically parses structured equity signals before the
Agent ON/OFF decision gate. The normalized intent is shown in the configured
review channel, including nested fields such as bracket exits, scale-in rules,
time-in-force, percentage closes, and conditional triggers.

## Execution behavior

- Agent OFF follows the incoming direction after validation.
- Agent ON runs the existing stock prediction gate; HOLD or a conflicting
  direction prevents submission.
- Market, limit, stop, stop-limit, market-on-open, market-on-close, notional,
  fractional, short, and buy-to-cover instructions map to Alpaca paper-order
  payloads.
- Percentage exits are calculated from the current Alpaca position.
- Take-profit and stop-loss pairs are submitted as Alpaca bracket orders when
  they can be enforced atomically.
- Multi-stage policies that require future monitoring are retained in the plan
  and blocked from partial submission until a durable monitor can enforce every
  leg. The agent never silently discards those fields.

## Corpus regression

`test_stock_order_intent.py` automatically reads the 300-case JSON corpus from
`TRADING_AGENT_TEST_CASES_JSON`, a local `test_data` folder, or the user's
Downloads folder. It checks all 120 stock cases and the stock-specific malformed
cases, then verifies Alpaca payload construction and Agent ON/OFF gating.

Run:

```powershell
python -m discord_stock_prediction_agent.test_stock_order_intent
```
