# Statistical Arbitrage: Mean-Reversion Pairs Trading Backtester

A runnable Python research project with a Streamlit dashboard, training-only
statistical estimates, a held-out backtest, and an explicit two-leg cash ledger.
Built for explaining a research process, including an unsuccessful hypothesis.
There is no brokerage integration or live execution.

**Tested on macOS with Python 3.13.17.** This version targets your installed Python
3.13 rather than requiring a separate Python 3.11 installation. Other Python
versions and operating systems were not verified. Package versions are pinned
in `requirements.txt`; `requirements-lock.txt` records the tested environment.

![Live Yahoo Finance dashboard, default cointegration filter enabled](outputs/dashboard_historical.png)

## Start here on a Mac

1. Unzip the project. Keep all files in the `pairs-trading-research` folder.
2. Open Visual Studio Code. Choose **File → Open Folder** and select that folder.
3. Choose **Terminal → New Terminal**. Paste these commands, one line at a time:

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The first line should show Python 3.13. If it shows another version, replace
`python3` in the first two lines with `python3.13`. The `.venv` folder is an
isolated project environment; this does not replace your system Python.

Open the **Local URL** printed in Terminal, normally
[http://localhost:8501](http://localhost:8501). Leave Terminal running. To stop,
press **Control+C** in Terminal. The first installation needs internet access.

For later sessions, open the project terminal and run:

```bash
source .venv/bin/activate
python -m streamlit run app.py
```

**Optional shortcut:** double-click `Start Research.command`. It creates the
environment, installs the pinned requirements, and starts the app. macOS may
ask you to permit opening a downloaded script. You can always use the Terminal
steps above instead; do not change system security settings.

In VS Code, use **Command+Shift+P → Python: Select Interpreter** and choose
`.venv/bin/python` if the editor reports missing packages.

## Use the dashboard

- The default pair is **SPY / IVV**; alternatives are **GLD / IAU**, **KO / PEP**,
  and a custom pair of Yahoo symbols. Ticker order defines the OLS regression.
- Change sidebar controls and click **Run research →** to apply them.
- **Yahoo Finance** downloads current adjusted daily closes. It never silently
  substitutes synthetic data or a stale snapshot when a request fails.
- **Bundled historical snapshot** loads the genuine, checksum-verified downloads
  included for the three preset pairs, ending November 6, 2026. This works
  without market-data internet access after dependencies are installed. It is
  explicitly labeled as a stored snapshot and does not refresh itself.
- **Synthetic demonstration** generates an artificial random walk with a
  mean-reverting residual. Every view prominently labels it artificial. Ticker
  names have no financial meaning in this mode; dates are artificial weekdays.
  Lookback controls its length; the end-date control is not used.
- The initial view is **Out of sample · 40%**. The **In sample · 60%** view is a
  fitted diagnostic with an explicit estimation-hindsight warning.
- Use **Pair diagnostics** for prices and spread, **Trades & accounting** for
  the complete metrics, completed trades, and downloadable daily ledger, and
  **Methodology** for assumptions and equations.

**Why are there zero trades by default?** The recorded SPY/IVV training test
has p ≈ 0.0600, which fails the specified p < 0.05 filter. Remaining in cash is
the intended behavior. For an explicitly exploratory view, uncheck **Suppress
trades if cointegration fails** and apply. That does not make a failed pair
qualified. Synthetic mode demonstrates trades without claiming market returns.

## Motivation and hypothesis

Two economically related securities can share a long-run price relationship.
If deviations from that relationship are stationary and sufficiently persistent,
a trader can hypothesize that buying the relatively cheap leg and shorting the
relatively expensive leg will earn convergence P&L. Execution delay and costs can
eliminate that opportunity. This project tests that hypothesis rather than
assuming related securities form profitable pairs.

Price-level correlation measures contemporaneous co-movement. Two trending
series can have very high correlation without a stationary residual. Engle–Granger
tests the null of **no cointegration**, under its integration-order assumptions.
An observed p-value below 0.05 rejects that null at the stated level; it does not
prove a lasting relationship or profitability. This MVP assumes I(1) inputs
without a full ADF/KPSS integration-order study. Nearly collinear numerical
warnings make the test inconclusive and suppress qualification.

## Methodology

### Data and chronological split

`yfinance.download(auto_adjust=True, interval="1d")` supplies split/dividend-adjusted
daily closes. Dates are aligned across the pair. Nonpositive, infinite,
nonnumeric, missing, and duplicate observations are removed. No interpolation,
backfill, or forward-fill invents prices. The UI reports dropped rows. A minimum
of 100 valid aligned observations is required.

The first `floor(0.60 × N)` observations form the training sample. The remaining
40% is held out. At the default snapshot, there are **1,255 observations**:

- Training: **November 8, 2021–November 5, 2024**, 753 observations.
- Testing: **November 6, 2024–November 6, 2026**, 502 observations.

Live downloads conservatively exclude today's daily candle until 16:15 New York
time. Yahoo supplies observations for actual available sessions; this simple
cutoff is not a complete exchange calendar. The exact downloaded range appears
in the UI, and data more than five calendar days behind the requested exclusive
end are flagged as stale. Custom international markets need a different cutoff.

### OLS, cointegration, and spread

Fit on training observations only, with an intercept:

```text
A_t = alpha + beta × B_t + residual_t
S_t = A_t − fitted_alpha − fitted_beta × B_t
```

The hedge ratio is a **share ratio**, not a dollar weight. Engle–Granger is run
on the training price levels with a constant and AIC lag selection. Correlation,
intercept, hedge ratio, and half-life also use training data only. Parameters are
frozen for the entire test sample. No pair ranking or parameter optimizer uses
the holdout. Manually switching among pairs after inspecting results can still
create selection bias; all three predetermined pairs are disclosed below.

The trailing z-score uses a configurable window (default 60 aligned observations):

```text
z_t = (S_t − trailing_mean_t) / trailing_sample_standard_deviation_t
```

The window includes close t and the previous `window−1` observations, which are
known when the signal is generated after close t. There are no centered windows
or future shifts. No entry occurs before the window is complete or when its
standard deviation is zero. Out-of-sample rolling windows can include earlier
training observations. Each sample starts flat; training trades and pending
training orders do not carry into testing.

Half-life comes from a training residual AR(1) regression with an intercept:

```text
S_t = c + phi × S_(t−1) + error_t
half_life = −log(2) / log(phi)
```

Report it as undefined unless `0 < phi < 1` and estimation is well-defined. It is
a noisy model estimate, not a forecast of the duration of an individual trade.

### Trading rules

| State | Decision after close t |
|---|---|
| Flat | Enter long spread if `−stop < z_t < −entry` |
| Flat | Enter short spread if `entry < z_t < stop` |
| Long spread | Exit if `z_t >= −exit`, or adverse stop `z_t <= −stop` |
| Short spread | Exit if `z_t <= exit`, or adverse stop `z_t >= stop` |
| Invalid signal | Stay flat, or request an exit if already active |

Defaults: entry 2, exit 0, adverse stop 3.5. Entry inequalities are strict; exits
are inclusive. A position that crosses zero is closed. A favorable jump across
the opposite entry band closes the old position without immediately reversing.
An entry is blocked at or outside the stop band. There is only one position at a
time. Positive beta is required for opposing legs; nonpositive beta is explicitly
labeled unsupported and trades are suppressed even if the gate is disabled.

### Execution and portfolio accounting

Signals at close t execute at the **next available aligned session's adjusted
close**, t+1. This is a deliberately simple daily execution proxy. A decision does
not earn the return from t to its t+1 execution. An old holding does earn P&L up
to its exit close. New holdings begin earning P&L only after that close. Last-row
signals with no subsequent execution session are ignored.

Each daily row follows this order:

1. Mark existing quantities to today's close and calculate both-leg market P&L.
2. Execute the previous session's desired state; rebalance active positions daily.
3. Debit fees on the actual absolute dollars changed in both legs.
4. Record cash, quantities, equity, exposures, turnover, fees, and net P&L.

For direction `d` (+1 long, −1 short), positive beta, and after-cost equity E:

```text
q_A = d × E / (A + beta × B)
q_B = −beta × q_A
weight_A = d × A / (A + beta × B)
weight_B = −d × beta × B / (A + beta × B)
abs(weight_A) + abs(weight_B) = 1
cost = (cost_bps / 10,000) × (abs(delta_q_A) × A + abs(delta_q_B) × B)
equity = cash + q_A × A + q_B × B
```

Starting capital is $100,000. A scalar numerical solve accounts for fees before
setting 100% gross exposure. Fractional units are allowed. Close-time sizing uses
the execution close and current equity; this idealized NAV sizing is not a
market-on-close order simulator. Beta remains fixed while position size changes.
The engine charges entry, exit, and every rebalance. At flat positions all equity
is cash, with zero interest. Short proceeds are tracked in cash; there is no
assumption that cash must equal starting equity while a spread is open.

There is no forced terminal exit. Open final positions remain marked to market;
their entry and rebalance costs are included, future liquidation costs are not.
They are excluded from completed-trade win rate and average holding period.
Completed trade P&L plus any open trade contribution reconciles to the change
in strategy equity. A nonpositive equity path stops with an explicit error;
bankruptcy and margin calls are not modeled.

### Performance definitions

- Total net return: ending equity / initial equity − 1.
- Annualized return: `(ending / initial) ** (252 / observed_return_intervals) − 1`.
- Annualized volatility: sample standard deviation of daily net returns × √252.
- Sharpe: daily mean / daily sample standard deviation × √252, **zero risk-free rate**.
- Maximum drawdown: minimum equity / running peak − 1, including initial capital.
- Completed trades, win rate (`net_pnl > 0`), and average aligned holding sessions.
- Dollar turnover: sum of both legs' absolute traded dollars, including rebalancing.
  Reported turnover multiple divides this by starting capital; it is neither halved
  nor annualized. Costs are the sum of all daily fees.

Zero-volatility Sharpe is **N/A**, and no-completed-trade win rate and holding
period are **N/A**. An all-cash zero return is measured behavior, not an invented
trade statistic. The first row begins in cash and is excluded from return
annualization. Missing sessions reduce the observation count; the 252 convention
then becomes approximate.

## Recorded historical results

Actual Yahoo downloads on **November 7, 2026**, through **November 6, 2026**.
Same five-year range, 60/40 split, default thresholds, and 5 bps per dollar
traded for all three predetermined pairs:

| Pair | Training correlation | Training p-value | OLS beta | Half-life, sessions | Qualified? | OOS completed trades | OOS net return |
|---|---:|---:|---:|---:|---|---:|---:|
| SPY / IVV | 0.999998 | 0.060015 | 0.993578 | 1.45 | No | 0 | 0.00% |
| GLD / IAU | 0.999982 | 0.236319 | 4.856711 | 6.08 | No | 0 | 0.00% |
| KO / PEP | 0.538112 | 0.841426 | 0.286774 | 159.49 | No | 0 | 0.00% |

All three fail the training filter, so all remain in cash with zero fees and
undefined Sharpe. This is the main result, not evidence of profitability. The
project retains the requested defaults rather than searching for a passing pair.
Exact reports, price data, features, and ledgers are in `outputs/historical_*`.
Future Yahoo adjustments or software changes may alter estimates.

**Explicitly exploratory SPY/IVV run, qualification gate disabled:** the same
held-out sample produced 20 completed trades, **−1.99% net return**, Sharpe
**−4.33**, maximum drawdown **−1.99%**, and **$1,989.34** in modeled costs.
No final position remained open. This run verifies accounting under real prices;
the pair still fails qualification and this is not a validated trading strategy.
See `outputs/exploratory_spy_ivv/report.json` and the accompanying trade ledger.
Costs are economically significant for tiny deviations in closely tracking ETFs.

The independent `outputs/synthetic_verified` run is for functional verification
only. Its returns are artificial and must never be cited as market performance.

## Verification

Run the automated suite after installation:

```bash
python -m pytest -q
```

See [TEST_REPORT.md](TEST_REPORT.md) for the recorded result, environment,
coverage, browser checks, corrected development failures, and limitations.
Tests verify the next-session delay with hand-calculated P&L, future-data
perturbation invariance, training-only estimates, cost sensitivity, both-leg
fees, exposure/cash reconciliation, missing data, no trades, stops, negative
beta, historical snapshots, and Streamlit form/view behavior.

The actual Streamlit server was started and checked in headless Chrome. These
are genuine captures of the running application:

![Actual historical price and spread diagnostics](outputs/dashboard_diagnostics.png)

![Artificial-data dashboard; the displayed returns are synthetic](outputs/dashboard_synthetic.png)

## Reproduce or export without the dashboard

```bash
# Fresh download; latest completed available session
python run_research.py --data live --tickers SPY IVV

# Exact historical snapshot and recorded end date; no market-data network needed
python run_research.py --data snapshot --tickers SPY IVV --end 2026-11-06

# Explicitly exploratory real-data accounting demonstration, gate disabled
python run_research.py --data snapshot --end 2026-11-06 --allow-unqualified --output outputs/run_exploratory

# Clearly labeled artificial demo
python run_research.py --data synthetic --output outputs/run_demo
```

The command writes `report.json`, adjusted prices, spread/z-scores, both samples'
daily ledgers and completed trades, and a standalone interactive equity HTML
chart. A failed download writes `failure.json` and returns an error code; it
does not create substitute market returns. CLI flags also support `--years`,
`--cost-bps`, and `--output`. UI controls provide the full threshold configuration.

## Project map

```text
pairs-trading-research/
├── app.py                     # Streamlit dashboard
├── data_loader.py             # Yahoo download, cleaning, snapshots, artificial demo
├── statistics.py              # OLS, Engle–Granger, half-life, spread, trailing z-score
├── strategy.py                # Validated parameters and single-position state machine
├── backtest.py                # Lagged execution, holdings/cash/fees, metrics, trade ledger
├── research.py                # Shared chronological pipeline
├── run_research.py            # CLI and JSON/CSV/interactive HTML exports
├── requirements.txt           # Pinned direct dependencies
├── requirements-lock.txt      # Full tested dependency environment, excluding browser tooling
├── Start Research.command     # Optional Mac launcher
├── .streamlit/config.toml     # Dark theme
├── data/                      # Genuine historical snapshots and SHA-256 provenance manifest
├── tests/                     # Automated verification
├── outputs/                   # Recorded reports, ledgers, charts, browser proof, screenshots
├── PORTFOLIO_NOTES.md          # Resume sentence and 30-second interview explanation
└── TEST_REPORT.md
```

`statistics.py` is a project module with the requested name and shadows Python's
standard-library `statistics` module inside this folder. This project uses NumPy
for basic statistics. If extending it with libraries that need the standard
module, rename it to `pair_statistics.py` and update its imports.

## Limitations and next research steps

Historical cointegration can break. A significant test does not guarantee
out-of-sample convergence. Hedge ratios are estimated with error, residuals can
change regime, and an AR(1) half-life can be misleading. A single chronological
holdout is not robust walk-forward validation, and repeated dashboard exploration
is not independent testing. Use a new untouched sample after exploration.

This strategy seeks to reduce shared market exposure through opposing legs.
Equal gross exposure and an OLS share hedge do **not** guarantee dollar neutrality,
market-beta neutrality, sector neutrality, or balanced factor exposures.

Adjusted-price units are a convenient total-return approximation rather than
tradable unadjusted shares. There are no separate dividend cash flows; adding
them again would double-count adjustments. Adjustment factors are obtained today
and can be revised, so this is not a point-in-time corporate-action database.
Results exclude borrow fees/availability, short rebates, financing, taxes, bid/ask
spreads beyond the generic fee assumption, market impact, margin and execution
uncertainty. Delayed stops cannot guarantee a maximum loss. Daily close data
hide intraday paths. Custom securities are assumed to be comparable USD assets;
currency conversion and exchange-calendar alignment are not implemented.

Useful future improvements are point-in-time corporate actions and raw execution
prices, integration-order diagnostics, walk-forward estimation, sensitivity
analysis confined to training data, factor-exposure checks, and realistic borrow,
financing, and execution models. None are presented as already implemented.

Yahoo/yfinance data access can be rate-limited and is subject to the provider's
terms. The bundled snapshots are for this personal educational research artifact;
review data redistribution rights before publishing data publicly. You can remove
`data/` and recorded CSV/JSON/HTML outputs when sharing a source-only repository
and use live downloads instead; the snapshot controls will then report unavailable.

## Put the project on GitHub manually

No GitHub account connection is required to run this project. No repository was
created or published for you.

1. Sign in at [GitHub](https://github.com) and click **New repository**.
2. Name it `pairs-trading-research`; start with a private repository while reviewing
   the artifact and data-sharing terms. Do not initialize a second README.
3. Click **creating a new file** or **uploading an existing file**, then drag the
   contents of this project folder into the upload area. Keep `.streamlit/config.toml`
   and `.gitignore`; hidden files may need to be revealed with **Command+Shift+.**
4. Upload source, tests, documentation, and screenshots. Exclude `.venv`, `.cache`,
   `.pytest_cache`, and `__pycache__`. The provided `.gitignore` covers those if
   using Git. For a public source-only repository, omit the downloaded datasets
   and their generated CSV/JSON/HTML outputs as described above.
5. Click **Commit changes**. Check that the README and images render correctly.
6. Add the repository link to your resume after you can explain the assumptions
   and demonstrate the app. Suggested wording is in `PORTFOLIO_NOTES.md`.

## Primary references

- [statsmodels: Engle–Granger `coint`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html)
- [yfinance: download options, adjusted OHLC and exclusive end date](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html)
- [Streamlit: native application testing](https://docs.streamlit.io/develop/concepts/app-testing)
