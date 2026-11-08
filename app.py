"""Run with: python -m streamlit run app.py"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_loader import DataError, completed_session_end, download_prices, snapshot_prices, synthetic_prices, validate_tickers
from research import run_research
from strategy import StrategyConfig


st.set_page_config(page_title="Pairs Lab | Statistical Arbitrage", page_icon="↗", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500;600;700&display=swap');
.stApp {background:#0b1019;color:#e6edf6;font-family:Inter,sans-serif}
.block-container {padding-top:4.3rem;padding-bottom:2rem;max-width:1600px}
[data-testid="stSidebar"] {background:#101824;border-right:1px solid #253044}
h1 {font-size:2.45rem!important;letter-spacing:-.065rem;font-weight:700!important}
h2,h3 {letter-spacing:-.025rem}
[data-testid="stMetric"] {background:#111c2b;border:1px solid #263349;border-radius:10px;padding:16px 18px;min-height:106px}
[data-testid="stMetricLabel"] {color:#94a6be;font-size:.78rem;text-transform:uppercase;letter-spacing:.06em}
[data-testid="stMetricValue"] {font-family:'IBM Plex Mono',monospace;font-size:1.65rem}
.eyebrow {font-family:'IBM Plex Mono',monospace;color:#4ae2be;letter-spacing:.15em;font-size:.74rem;margin-bottom:.6rem}
.subtitle {color:#91a4bd;max-width:950px;margin-top:-.45rem;margin-bottom:1rem;font-size:.93rem}
.status {border-left:3px solid #4ae2be;background:#122331;padding:13px 16px;border-radius:0 6px 6px 0;color:#c9e8e3;margin:15px 0}
.small-note {color:#91a4bd;font-size:.79rem}
button[kind="primary"] {background:#37cba8;color:#071913;border:0;font-weight:600}
[data-baseweb="tab-list"] {gap:22px;border-bottom:1px solid #263349}
[data-baseweb="tab"] {font-size:.9rem}
footer {visibility:hidden}
</style>""", unsafe_allow_html=True)

COLORS = {"green": "#4ae2be", "blue": "#72a7ff", "amber": "#f8bf65", "red": "#f47f95"}


def figure(title: str, ytitle: str = "", height: int = 330) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#dce7f5")), height=height,
        template="plotly_dark", paper_bgcolor="#0b1019", plot_bgcolor="#0b1019",
        margin=dict(l=12, r=12, t=45, b=20), font=dict(family="Inter, sans-serif", color="#9caec6"),
        legend=dict(orientation="h", y=1.13, x=1, xanchor="right"),
        hovermode="x unified", yaxis_title=ytitle,
        xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#202c3e", zerolinecolor="#35445b"),
    )
    return fig


def line(fig, series, name, color, **kwargs):
    fig.add_trace(go.Scatter(x=series.index, y=series, name=name, mode="lines",
                             line=dict(color=color, width=1.8), **kwargs))


def pct(value):
    return "N/A" if value is None else f"{value:+.2%}"


def num(value, digits=2):
    return "N/A" if value is None else f"{value:.{digits}f}"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch(a, b, start, end):
    return download_prices(a, b, start, end)


@st.cache_data(show_spinner=False)
def analyze(prices, config):
    return run_research(prices, config)


with st.sidebar:
    st.markdown("### ◈ PAIRS LAB")
    st.caption("STATISTICAL ARBITRAGE / RESEARCH")
    with st.form("research_controls"):
        st.markdown("#### Research universe")
        choice = st.selectbox("Ticker pair", ["SPY / IVV", "GLD / IAU", "KO / PEP", "Custom pair"])
        custom_a = st.text_input("Custom ticker A", "SPY")
        custom_b = st.text_input("Custom ticker B", "IVV")
        demo = os.environ.get("PAIRS_DEMO") == "1" or st.query_params.get("demo") == "1"
        snapshot = st.query_params.get("snapshot") == "1"
        source = st.radio("Data source", ["Yahoo Finance", "Synthetic demonstration", "Bundled historical snapshot"],
                          index=2 if snapshot else int(demo), key="data_source")
        years = st.slider("Lookback (years)", 1, 10, 5)
        end = st.date_input("Data through (inclusive)", completed_session_end() - timedelta(days=1),
                            max_value=completed_session_end() - timedelta(days=1),
                            disabled=source == "Synthetic demonstration",
                            help="Synthetic mode uses fixed artificial weekdays, independent of this date.")
        st.markdown("#### Signal & execution")
        entry = st.slider("Entry threshold |z|", 0.5, 3.0, 2.0, 0.1)
        exit_level = st.slider("Exit threshold |z|", 0.0, 1.0, 0.0, 0.1)
        stop = st.slider("Adverse stop |z|", 2.5, 6.0, 3.5, 0.1)
        window = st.slider("Rolling window (observations)", 10, 180, 60, 5)
        costs = st.number_input("Costs (bps per dollar traded)", 0.0, 100.0, 5.0, 0.5)
        gate = st.checkbox("Suppress trades if cointegration fails", value=True, key="qualification_gate")
        st.form_submit_button("Run research →", type="primary", width="stretch", key="run_research")
    if st.button("Refresh Yahoo data", width="stretch"):
        fetch.clear()
        st.rerun()
    st.caption("$100,000 initial equity · 100% gross exposure\n\nSignals at close → next session close execution")

try:
    a, b = validate_tickers(*(choice.split(" / ") if choice != "Custom pair" else (custom_a, custom_b)))
    config = StrategyConfig(entry, exit_level, stop, int(window), float(costs), require_cointegration=gate)
    end_exclusive = end + timedelta(days=1)
    start = (pd.Timestamp(end_exclusive) - pd.DateOffset(years=years)).date()
    with st.spinner("Aligning prices, fitting the training sample, and simulating both legs…"):
        if source == "Synthetic demonstration":
            bundle = synthetic_prices(a, b, observations=years * 252)
        elif source == "Bundled historical snapshot":
            bundle = snapshot_prices(a, b, start, end_exclusive)
        else:
            bundle = fetch(a, b, start, end_exclusive)
        prices = bundle.prices
        result = analyze(prices, config)
except (DataError, ValueError, KeyError) as exc:
    st.title("Statistical Arbitrage")
    st.error(str(exc))
    st.info("Check your internet connection and retry, or select Bundled historical snapshot or Synthetic demonstration in the sidebar and click Run research. Synthetic mode never displays market performance.")
    st.stop()

model, features = result.model, result.features
st.markdown('<div class="eyebrow">QUANTITATIVE RESEARCH &nbsp; / &nbsp; PROJECT 01</div>', unsafe_allow_html=True)
st.title("Statistical Arbitrage")
st.markdown(f'<div class="subtitle">Mean-reversion pairs backtester &nbsp; · &nbsp; <b>{a} / {b}</b> &nbsp; · &nbsp; '
            f'{prices.index[0]:%d %b %Y} — {prices.index[-1]:%d %b %Y} &nbsp; · &nbsp; {len(prices):,} aligned observations</div>',
            unsafe_allow_html=True)
if bundle.synthetic:
    st.warning("SYNTHETIC DEMONSTRATION — artificial prices and returns. Ticker names are labels; these are not historical market results.")
st.caption(f"Source: {bundle.source}. {bundle.note} Dropped rows: {bundle.dropped_rows}.")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Training cointegration p", f"{model.pvalue:.4g}")
c2.metric("OLS hedge ratio β", f"{model.beta:.6f}")
c3.metric("Residual half-life", "Undefined" if model.half_life is None else f"{model.half_life:.1f} sessions")
c4.metric("Training correlation", f"{model.correlation:.6f}")
st.markdown(f'<div class="status">● &nbsp; {result.status}</div>', unsafe_allow_html=True)
if model.warning:
    st.warning(model.warning + " An unstable test is not treated as a pass.")
if not model.qualified and not gate:
    st.warning("Exploratory trading is enabled for a pair that has not passed the training cointegration test.")

view = st.radio("Evaluation sample", ["Out of sample · 40%", "In sample · 60%"], horizontal=True, key="evaluation")
oos = view.startswith("Out")
section = slice(result.split, None) if oos else slice(None, result.split)
sample, f = prices.iloc[section], features.iloc[section]
bt = result.out_of_sample if oos else result.in_sample
m, daily = bt.metrics, bt.daily
if not oos:
    st.warning("IN SAMPLE / FITTED DIAGNOSTIC: the model uses this entire training interval, so these fitted results contain estimation hindsight. Use the held-out view for evaluation.")
st.caption(f"Training: {prices.index[0]:%Y-%m-%d} → {model.train_end:%Y-%m-%d} · "
           f"Testing: {prices.index[result.split]:%Y-%m-%d} → {prices.index[-1]:%Y-%m-%d}")
p1, p2, p3, p4 = st.columns(4)
p1.metric("Net total return", pct(m["total_return"]))
p2.metric("Sharpe · rf = 0", num(m["sharpe"]))
p3.metric("Maximum drawdown", f'{m["max_drawdown"]:.2%}')
p4.metric("Completed trades", str(m["completed_trades"]))

performance, diagnostics, ledger, method = st.tabs(["Strategy performance", "Pair diagnostics", "Trades & accounting", "Methodology"])
with performance:
    equity = figure("Cumulative strategy equity", "USD", 340)
    line(equity, daily.equity, "Net equity", COLORS["green"])
    equity.add_hline(y=config.initial_capital, line_dash="dot", line_color="#5c6d86")
    lower = min(float(daily.equity.min()), config.initial_capital)
    upper = max(float(daily.equity.max()), config.initial_capital)
    padding = max((upper - lower) * 0.15, config.initial_capital * 0.005)
    equity.update_yaxes(range=[lower - padding, upper + padding])
    st.plotly_chart(equity, width="stretch")
    left, right = st.columns([1.65, 1])
    with left:
        zfig = figure("Z-score & executed trades", "Rolling z", 310)
        line(zfig, f.zscore, "Z-score", COLORS["blue"])
        for threshold, color, dash in [(entry, "amber", "dash"), (-entry, "amber", "dash"),
                                       (stop, "red", "dot"), (-stop, "red", "dot"),
                                       (exit_level, "green", "dot"), (-exit_level, "green", "dot")]:
            zfig.add_hline(y=threshold, line_color=COLORS[color], line_dash=dash, opacity=0.65)
        for event, symbol, color in [("entry", "triangle-up", "green"), ("exit", "x", "red")]:
            dates = daily.index[daily.event == event]
            zfig.add_trace(go.Scatter(x=dates, y=f.zscore.reindex(dates), mode="markers", name=event.title(),
                                     marker=dict(symbol=symbol, size=9, color=COLORS[color]),
                                     text=daily.loc[dates, "signal_date"].astype(str),
                                     hovertemplate="Executed %{x}<br>z at execution: %{y:.2f}<br>Signal date: %{text}<extra></extra>"))
        st.plotly_chart(zfig, width="stretch")
    with right:
        drawdown = figure("Drawdown", "Below equity peak", 310)
        line(drawdown, daily.drawdown, "Drawdown", COLORS["red"], fill="tozeroy", fillcolor="rgba(244,127,149,0.13)")
        drawdown.update_yaxes(tickformat=".1%", range=[min(float(daily.drawdown.min()) * 1.1, -0.001), 0.00005])
        st.plotly_chart(drawdown, width="stretch")
    st.caption("Markers show execution dates, one aligned session after the decision. Equity includes all modeled entry, exit, and daily rebalancing fees. Open final positions are marked to market.")

with diagnostics:
    normalized = st.checkbox("Normalize both price series to 100", True)
    pfig = figure("Adjusted daily prices · full research window", "Index = 100" if normalized else "Adjusted USD", 380)
    chart_prices = prices / prices.iloc[0] * 100 if normalized else prices
    for ticker, color in zip((a, b), ("green", "blue")):
        line(pfig, chart_prices[ticker], ticker, COLORS[color])
    pfig.add_vline(x=prices.index[result.split].timestamp() * 1000, line_dash="dash", line_color=COLORS["amber"])
    st.plotly_chart(pfig, width="stretch")
    st.caption("Vertical line: beginning of the held-out test sample. All model estimates come from the earlier 60%.")
    sfig = figure("OLS residual spread · full research window", "Adjusted-price residual", 350)
    line(sfig, features.spread, "Spread", COLORS["blue"])
    line(sfig, features.rolling_mean, "Trailing mean", COLORS["green"])
    sfig.add_vline(x=prices.index[result.split].timestamp() * 1000, line_dash="dash", line_color=COLORS["amber"])
    st.plotly_chart(sfig, width="stretch")
    st.write(f"Training fit: {a} = {model.alpha:.6f} + {model.beta:.6f} × {b} + residual.")
    st.caption("Price-level correlation is descriptive; high correlation alone does not establish a stationary spread. Engle–Granger assumes I(1) input series; this MVP does not establish that assumption with a full integration-order study.")

with ledger:
    c1, c2, c3 = st.columns(3)
    c1.metric("Annualized return", pct(m["annualized_return"]))
    c2.metric("Annualized volatility", "N/A" if m["annualized_volatility"] is None else f'{m["annualized_volatility"]:.2%}')
    c3.metric("Win rate · completed", "N/A" if m["win_rate"] is None else f'{m["win_rate"]:.1%}')
    c1, c2, c3 = st.columns(3)
    c1.metric("Average holding period", "N/A" if m["average_holding_sessions"] is None else f'{m["average_holding_sessions"]:.1f} sessions')
    c2.metric("Turnover / starting capital", f'{m["turnover_multiple"]:.2f}×')
    c3.metric("Total modeled costs", f'${m["total_costs"]:,.2f}')
    st.subheader("Completed trades")
    if bt.trades.empty:
        st.info("No completed trades in this sample. Win rate and average holding period are undefined.")
    st.dataframe(bt.trades, hide_index=True, width="stretch")
    if bt.open_trade:
        st.info(f'One final position remains open: {bt.open_trade["side"]}, entered {bt.open_trade["entry_date"]:%Y-%m-%d}, net unrealized contribution ${bt.open_trade["net_pnl"]:,.2f}. It is excluded from completed-trade statistics.')
    st.download_button("Download completed trades CSV", bt.trades.to_csv(index=False), f"{a}_{b}_trades.csv", "text/csv")
    st.subheader("Daily accounting ledger")
    st.dataframe(daily.tail(100), width="stretch")
    st.download_button("Download full daily ledger CSV", daily.to_csv(), f"{a}_{b}_daily_ledger.csv", "text/csv")
    st.caption("Gross turnover sums absolute dollars traded across both legs, including every rebalance, divided by starting capital. It is not halved or annualized. Holding periods count aligned observations.")

with method:
    st.markdown("""
### Research hypothesis
Related securities may share a long-run price relationship. A temporary deviation
from that relationship can offer a mean-reversion hypothesis to test.

1. Align adjusted daily closes, dropping missing or invalid rows without filling prices.
2. Split chronologically: the first 60% fits OLS and tests Engle–Granger cointegration;
   the last 40% is held out. A p-value below 0.05 rejects the no-cointegration null.
3. Freeze the intercept and hedge ratio for the entire held-out period.
4. Compute a trailing spread mean and sample standard deviation, including today's close.
5. Enter beyond ±entry but inside the stop band. Exit when a long spread rises to
   −exit or a short spread falls to +exit. A move further against the trade to
   −stop (long) or +stop (short) requests an exit. At the default exit of zero,
   crossing the mean exits the position. No overlapping positions or same-day reversal.
6. Execute that decision at the **next aligned session's adjusted close**. Existing
   holdings earn P&L up to that execution; new holdings start earning P&L afterward.
""")
    st.latex(r"S_t = A_t - \hat\alpha - \hat\beta B_t,\qquad z_t = \frac{S_t-\bar S_{t,w}}{s_{t,w}}")
    st.latex(r"q_{A,t} = \frac{d_t E_t}{A_t+\hat\beta B_t},\quad q_{B,t}=-\hat\beta q_{A,t},\quad d_t\in\{-1,0,1\}")
    st.markdown("""
Daily rebalancing preserves the fitted **share ratio**, with gross exposure equal
to 100% of after-cost equity. Fees apply to the actual absolute dollar change in
both legs. Cash earns zero. Sharpe assumes zero risk-free rate; annualization uses
252 aligned observations per year. Half-life is −log(2)/log(φ) from a training
AR(1) residual model, defined only for 0 < φ < 1.

**Limits:** This is a research prototype. Adjusted closes are a total-return proxy,
not executable quotes; adjustments can be revised. No short borrow, financing,
separate dividends, taxes, bid/ask model, market impact, margin, or availability
checks are included. A lagged stop is not a guaranteed loss limit. Short-sale
proceeds remain in the cash ledger. Open trades are not forcibly liquidated.
Missing aligned sessions can delay executions and distort 252-day annualization.

In-sample results use a model fitted on that whole interval and contain hindsight.
Repeatedly changing pairs or thresholds after seeing held-out performance turns
the holdout into a selection sample. Freeze your hypothesis before evaluation.
Historical cointegration can break. This strategy aims to reduce common market
exposure but is not necessarily dollar-neutral or beta-neutral, and does not
guarantee a profit.
""")
st.caption("PAIRS LAB  /  Historical research only · No brokerage connection or live trading")
