# Portfolio and interview notes

## Resume description

Built a Python statistical-arbitrage research dashboard using OLS, Engle–Granger
cointegration, and trailing spread z-scores, with a chronological holdout,
next-session execution, two-leg transaction-cost accounting, and automated tests;
evaluated SPY/IVV, GLD/IAU, and KO/PEP on genuine historical data.

Do not claim profitable trading, live execution, or synthetic returns as market
performance. The default historical tests did not qualify the three preset pairs.

## 30-second interview explanation

“I tested whether deviations between related securities revert toward an
estimated long-run relationship. I fit the hedge ratio and tested cointegration
on the first 60% of the data, then evaluated frozen parameters on the remaining
40%. Signals trade one session later, with costs on both legs and rebalancing.
All three preset pairs failed the training filter, so the default strategy stayed
in cash. An exploratory run showed how small apparent ETF deviations can be
overwhelmed by transaction costs.”

## A short demonstration

1. Open the historical SPY/IVV view. Explain why its p ≈ 0.0600 fails the 0.05
   requirement despite correlation above 0.99999.
2. Open Pair diagnostics. Explain the training/test boundary and frozen OLS beta.
3. Show the zero-trade, all-cash result as a valid research finding.
4. Disable the qualification filter explicitly for an exploratory accounting
   demonstration. Point out the warning, negative net performance, and fees.
5. Open Trades & accounting and show a signal date followed by a later execution
   date. Explain cash, leg quantities, and turnover.
6. Switch to Synthetic demonstration only to illustrate a designed mean-reverting
   process. Explicitly say its prices and returns are artificial.

## Questions to be ready for

- Correlation describes co-movement; cointegration concerns stationarity of a
  linear combination of integrated price series.
- The OLS beta is a share hedge. Gross normalization does not imply dollar or
  market-beta neutrality.
- Close-t signals cannot earn P&L before the next aligned closing execution.
- In-sample fitted results contain estimation hindsight and are not a prospective
  performance claim.
- Daily rebalancing adds turnover. Short borrow and financing can worsen results
  further; they are not included in this prototype.
- A failed hypothesis can demonstrate disciplined research. The threshold was
  not weakened to manufacture a qualified default strategy.
