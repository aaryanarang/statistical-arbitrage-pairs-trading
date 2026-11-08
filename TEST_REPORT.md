# Executed verification report

Recorded November 7, 2026 (America/New_York). These are executed checks, not
promised future tests. This is a research prototype, not a production trading
certification.

## Environment and installation

- macOS / Apple Silicon, Python **3.13.17**.
- Installed NumPy 2.5.4, pandas 3.1.1, SciPy 1.18.1, statsmodels 0.15.0,
  yfinance 1.8.0, Plotly 7.1.0, Streamlit 1.67.0, pytest 9.2.0.
- Full tested dependency versions are in `requirements-lock.txt`.
- Browser verification used installed Google Chrome with Playwright. Browser
  tooling is not required to install or run the delivered application.
- The first package download was blocked by this agent's network restrictions.
  Installation succeeded with network permission. The local server similarly
  required permission to listen on localhost in the test environment. These
  restrictions are specific to the agent environment, not app requirements.

## Automated result

```text
$ python -m pytest -q
.............................................                            [100%]
45 passed in 4.78s
```

The raw output is saved in `outputs/pytest_results.txt`. Tests cover:

| Area | Executed verification |
|---|---|
| Strategy | Long/short entries, zero/band exits, adverse stops, warm-up, invalid signals, gate suppression, no immediate reversal, parameter validation |
| Causality | Changing a future suffix cannot change past signals, held-out holdings, P&L, trailing features, or the training fit |
| Execution | Hand-calculated two-leg P&L, next-aligned-close entry/exit, no P&L before entry execution, unexecuted final signals |
| Accounting | Beta share ratio, 100% after-fee gross exposure, cash/equity identities, completed plus open P&L reconciliation, daily rebalance fees |
| Costs | Controlled flat-price roundtrip and genuine SPY/IVV snapshot both worsen with fees |
| Data | Invalid/missing/duplicate rows removed without filling; adjusted download flags mocked; failure does not silently use artificial data |
| Statistics | Training-only OLS/cointegration, trailing sample-standard-deviation z-score, zero-variance handling, valid/undefined half-life |
| Edge cases | No trades, open terminal trade, unsupported nonpositive beta, missing prices rejected at engine boundary |
| Actual data | End-to-end pipeline runs on all three genuine historical snapshots |
| Dashboard | Synthetic render, in/out-of-sample switch, real-snapshot form submission and exploratory warning |

Early development checks found one syntax issue and UI automation that selected
the wrong control or clicked the hidden checkbox input. These were corrected;
controls now have stable test keys, and the browser check clicks their visible
labels. **No final automated test failures remain.**

## Actual historical data pipeline

Yahoo Finance downloads succeeded for SPY/IVV, GLD/IAU, and KO/PEP, each with
1,255 aligned daily adjusted observations from November 8, 2021 to November 6,
2026. The model uses 753 training observations; held-out results use 502.
Engle–Granger, OLS, half-life, spread, trailing z-score, signals, and the portfolio
engine all ran. Snapshots include SHA-256 hashes in `data/manifest.json`.

All three pairs failed p < 0.05 on the training sample. At the enabled default
gate the strategy correctly remained in cash. See README for exact p-values and
`outputs/historical_*/report.json` for complete results.

To exercise actual-price entries/exits as well as suppression, SPY/IVV was run
with qualification explicitly disabled and all other defaults unchanged. It
completed 20 held-out trades. This is marked **exploratory**, with no claim that
the unqualified pair meets the research hypothesis.

The same real prices and signals were also checked at 0, 5, and 25 bps. Position
sizes scale with net equity, so cost totals are not assumed to increase linearly.

| Cost, bps | Held-out ending equity | Modeled costs | Completed trades |
|---:|---:|---:|---:|
| 0 | $99,999.50 | $0.00 | 20 |
| 5 | $98,010.16 | $1,989.34 | 20 |
| 25 | $90,440.79 | $9,558.74 | 20 |

Exact sensitivity results are saved in `outputs/cost_sensitivity.json`. These
measure the specified accounting assumptions, not implementable trading returns.
The deterministic synthetic pipeline also ran, with every artifact labeled as
artificial. Synthetic performance is excluded from historical findings.

## Running server and actual browser

- Launched Streamlit on `127.0.0.1:8501`.
- The server health endpoint returned **ok**.
- Headless Chrome opened the default **live Yahoo** dashboard and rendered metrics
  and Plotly charts, with no Streamlit exception displayed.
- Opened Pair diagnostics and captured the adjusted prices and spread.
- Opened synthetic mode, verified the prominent artificial-data banner and charts.
- Opened bundled historical mode, disabled the gate using the visible checkbox,
  submitted the form, observed the exploratory warning and 20-trade result, and
  opened Trades & accounting.
- Browser JavaScript page errors: **none**.

Raw observations: `outputs/browser_verification.json`. Genuine screenshots:
`dashboard_historical.png`, `dashboard_diagnostics.png`, and
`dashboard_synthetic.png` in `outputs/`. Screenshots are actual browser captures;
they are not mockups. A fresh live request can differ slightly from the recorded
snapshot as provider adjustments change; qualification stayed failed in both.

## Scope and verification limits

- Tested Python 3.13 on this Mac; Python 3.11, other operating systems, different
  hardware, and other package combinations were not verified.
- The Mac launcher was syntax-checked; the tested installation and launch also
  used the same underlying Python/pip/Streamlit commands directly. A Finder
  double-click and macOS downloaded-file prompts were not automated.
- The offline historical pipeline, CLI, and dashboard were executed. Dependency
  installation still requires package access on a new computer.
- No brokerage, order routing, live execution, short availability, borrow/financing,
  margin, or point-in-time corporate-action data were tested or implemented.
- Statistical tests do not prove the relationship will persist. In-sample results
  use the model fitted over that same interval and have estimation hindsight.
- No repository was published and no GitHub authentication was attempted.

The delivered ZIP excludes virtual environments, caches, and bytecode. Its source,
tests, snapshots, documentation, screenshots, and recorded outputs are included.
The release archive passed its CRC check, and every extracted file matched its
source byte for byte. Running the suite from the extracted archive also passed:
**45 passed in 4.41s**. `pip check` reported **No broken requirements found**.
