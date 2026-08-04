import random
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Pro Backtest: Slippage & Real SL", layout="wide")
st.title("Quantitative Dashboard: Advanced MA Crossover")

# --- UI: SELECTION MENUS ---
st.subheader("Select Moving Average Type:")
ma_type = st.selectbox(
    "",
    [
        "SMA (Simple Moving Average)",
        "EMA (Exponential Moving Average)",
        "AAMA (Alvin Adaptive Moving Average)"
    ],
    label_visibility="collapsed"
)

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    strategy_choice = st.selectbox(
        "Select Strategy (Periods)",
        ["20 / 50 (Short/Medium Term)", "50 / 200 (Macro Trend)"]
    )

with col2:
    market_period = st.selectbox(
        "Select Market Regime",
        [
            "Bull Market (Jan 2020 - Dec 2021) - Post-Covid Euphoria",
            "Bear Market (Jan 2022 - Dec 2022) - FED Rate Hikes",
            "Volatile / Range (Jan 2018 - Dec 2019) - Uncertain Market"
        ]
    )

with col3:
    inflation_choice = st.selectbox(
        "Adjust for Inflation?",
        ["No (Nominal Returns)", "Yes (Real Returns)"]
    )

if "20 / 50" in strategy_choice:
    MA_FAST, MA_SLOW = 20, 50
else:
    MA_FAST, MA_SLOW = 50, 200

if "Bull Market" in market_period:
    START_DATE, END_DATE = "2020-01-01", "2021-12-31"
    INF_RATE = 0.03
elif "Bear Market" in market_period:
    START_DATE, END_DATE = "2022-01-01", "2022-12-31"
    INF_RATE = 0.08
else:
    START_DATE, END_DATE = "2018-01-01", "2019-12-31"
    INF_RATE = 0.021

APPLY_INFLATION = "Yes" in inflation_choice
INITIAL_CAPITAL = 10000
FEE_RATE = 0.001

MARKETS = {
    "Stocks": ["AAPL", "TSLA", "NVDA"],
    "Indices": ["^GSPC", "^NDX"],
    "Crypto": ["BTC-USD", "ETH-USD"]
}


@st.cache_data
def run_realistic_backtest(ticker, start, end, ma_f, ma_s, apply_inf, inf_rate, ma_type_str):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        return None, None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "EMA" in ma_type_str:
        df['MA_Fast'] = df['Close'].ewm(span=ma_f, adjust=False).mean()
        df['MA_Slow'] = df['Close'].ewm(span=ma_s, adjust=False).mean()
    elif "AAMA" in ma_type_str:
        lambda_param = 1.0

        # Fast MA
        base_alpha_fast = 2 / (ma_f + 1)
        rolling_std_fast = df['Close'].rolling(window=ma_f).std().fillna(0)
        aama_fast_vals = [float(df['Close'].iloc[0])]
        for i in range(1, len(df)):
            p = float(df['Close'].iloc[i])
            sig = float(rolling_std_fast.iloc[i])
            cv = (sig / p) if p != 0 else 0
            alpha_t = base_alpha_fast * np.exp(-lambda_param * cv)
            alpha_t = max(0.01, min(1.0, alpha_t))
            prev_val = aama_fast_vals[-1]
            aama_fast_vals.append(alpha_t * p + (1 - alpha_t) * prev_val)
        df['MA_Fast'] = aama_fast_vals

        # Slow MA
        base_alpha_slow = 2 / (ma_s + 1)
        rolling_std_slow = df['Close'].rolling(window=ma_s).std().fillna(0)
        aama_slow_vals = [float(df['Close'].iloc[0])]
        for i in range(1, len(df)):
            p = float(df['Close'].iloc[i])
            sig = float(rolling_std_slow.iloc[i])
            cv = (sig / p) if p != 0 else 0
            alpha_t = base_alpha_slow * np.exp(-lambda_param * cv)
            alpha_t = max(0.01, min(1.0, alpha_t))
            prev_val = aama_slow_vals[-1]
            aama_slow_vals.append(alpha_t * p + (1 - alpha_t) * prev_val)
        df['MA_Slow'] = aama_slow_vals
    else:
        df['MA_Fast'] = df['Close'].rolling(window=ma_f).mean()
        df['MA_Slow'] = df['Close'].rolling(window=ma_s).mean()

    discount_factors = ((1 + inf_rate) ** ((df.index - df.index[0]).days / 365.25)).to_numpy()

    capital = INITIAL_CAPITAL
    position = 0
    buy_price = 0
    stop_loss = 0

    total_trades = 0
    winning_trades = 0
    losing_trades = 0
    gross_profit = 0.0
    gross_loss = 0.0
    max_loss_trade = 0.0
    current_trade_duration = 0
    trade_durations = []

    portfolio_values = []
    win_rates = []

    for i in range(len(df)):
        if i < ma_s + 1:
            portfolio_values.append(capital)
            win_rates.append(np.nan)
            continue

        open_price = float(df['Open'].iloc[i])
        low_price = float(df['Low'].iloc[i])
        close_price = float(df['Close'].iloc[i])

        prev_fast = float(df['MA_Fast'].iloc[i - 1])
        prev_slow = float(df['MA_Slow'].iloc[i - 1])
        prev_prev_fast = float(df['MA_Fast'].iloc[i - 2])
        prev_prev_slow = float(df['MA_Slow'].iloc[i - 2])

        current_discount = float(discount_factors[i]) if apply_inf else 1.0

        slippage = open_price * random.uniform(0.0005, 0.0015)

        trade_closed = False
        trade_pnl_nominal = 0

        if position > 0:
            current_trade_duration += 1

            if low_price <= stop_loss:
                exec_price = stop_loss - slippage
                capital += position * exec_price * (1 - FEE_RATE)
                trade_pnl_nominal = (exec_price * (1 - FEE_RATE)) - buy_price
                position = 0
                trade_closed = True

            elif (prev_fast < prev_slow) and (prev_prev_fast >= prev_prev_slow):
                exec_price = open_price - slippage
                capital += position * exec_price * (1 - FEE_RATE)
                trade_pnl_nominal = (exec_price * (1 - FEE_RATE)) - buy_price
                position = 0
                trade_closed = True

            if trade_closed:
                total_trades += 1
                trade_durations.append(current_trade_duration)
                current_trade_duration = 0

                real_pnl = trade_pnl_nominal / current_discount

                if real_pnl > 0:
                    winning_trades += 1
                    gross_profit += real_pnl
                else:
                    losing_trades += 1
                    gross_loss += abs(real_pnl)

                if real_pnl < max_loss_trade:
                    max_loss_trade = real_pnl

        elif position == 0:
            if (prev_fast > prev_slow) and (prev_prev_fast <= prev_prev_slow):
                exec_price = open_price + slippage
                stop_loss_price = exec_price * 0.98  # SL at -2%

                risk_amount = capital * 0.01
                risk_per_share = exec_price - stop_loss_price

                qty = risk_amount / risk_per_share
                max_qty = capital / (exec_price * (1 + FEE_RATE))
                qty = min(qty, max_qty)

                cost = qty * exec_price * (1 + FEE_RATE)
                capital -= cost
                position = qty
                buy_price = exec_price * (1 + FEE_RATE)
                stop_loss = stop_loss_price

        current_val_nominal = capital + (position * close_price)
        portfolio_values.append(current_val_nominal)

        if total_trades > 0:
            current_win_rate = (winning_trades / total_trades) * 100
        else:
            current_win_rate = np.nan

        win_rates.append(current_win_rate)

    df['Nominal_Value'] = portfolio_values

    if apply_inf:
        df['Portfolio_Value'] = df['Nominal_Value'] / discount_factors
    else:
        df['Portfolio_Value'] = df['Nominal_Value']

    df['Win_Rate'] = win_rates

    stats_dict = {
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "max_loss": max_loss_trade,
        "trade_durations": trade_durations
    }

    return df, stats_dict


# Extraction du nom du type pour l'affichage
if "AAMA" in ma_type:
    ma_name_display = "AAMA (Alvin Adaptive MA)"
elif "EMA" in ma_type:
    ma_name_display = "EMA"
else:
    ma_name_display = "SMA"

st.header(f"Results for {ma_name_display} {strategy_choice} in {market_period.split('-')[0]}")

market_portfolio_values = {category: pd.DataFrame() for category in MARKETS.keys()}
all_portfolio_values = pd.DataFrame()
all_win_rates = pd.DataFrame()
all_stats_dict = {}

for market_name, tickers in MARKETS.items():
    st.subheader(f"{market_name}")

    for ticker in tickers:
        with st.expander(f"Details: {ticker}"):
            df, stats = run_realistic_backtest(ticker, START_DATE, END_DATE, MA_FAST, MA_SLOW, APPLY_INFLATION,
                                               INF_RATE, ma_type)

            if df is not None:
                market_portfolio_values[market_name][ticker] = df['Portfolio_Value']
                all_portfolio_values[ticker] = df['Portfolio_Value']
                all_win_rates[ticker] = df['Win_Rate']
                all_stats_dict[ticker] = stats

                fig = go.Figure()

                curve_name = 'Real Capital (Adjusted)' if APPLY_INFLATION else 'Capital'
                fig.add_trace(go.Scatter(x=df.index, y=df['Portfolio_Value'], mode='lines', name=curve_name,
                                         line=dict(color='#2980b9', width=2)))

                if APPLY_INFLATION:
                    fig.add_trace(
                        go.Scatter(x=df.index, y=df['Nominal_Value'], mode='lines', name='Nominal Capital (Unadjusted)',
                                   line=dict(color='#bdc3c7', width=1.5, dash='dot')))

                fig.add_hline(y=INITIAL_CAPITAL, line_dash="dot", line_color="red", annotation_text="Initial Capital")
                chart_title = f"Net Capital ({ticker}) - Real Value" if APPLY_INFLATION else f"Net Capital ({ticker}) - Nominal Value"

                fig.update_layout(title=chart_title, xaxis_title="Date", yaxis_title="Value ($)",
                                  template="plotly_white", height=300)
                st.plotly_chart(fig, width='stretch')

st.header("Global Portfolio Analysis")

selected_categories = st.multiselect(
    "Select markets to include in the Overall Portfolio Average:",
    options=list(MARKETS.keys()),
    default=list(MARKETS.keys())
)

active_tickers = [
    ticker for category in selected_categories 
    for ticker in MARKETS[category] 
    if ticker in all_portfolio_values.columns
]

if not active_tickers:
    st.warning("Please select at least one market category to display global analysis.")
else:
    df_filtered_portfolio = all_portfolio_values[active_tickers]
    df_filtered_winrates = all_win_rates[active_tickers]

    avg_portfolio = df_filtered_portfolio.mean(axis=1)
    avg_win_rate = df_filtered_winrates.mean(axis=1)

    dyn_global_stats = {
        "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
        "gross_profit": 0.0, "gross_loss": 0.0, "max_loss": 0.0, "trade_durations": []
    }

    for t in active_tickers:
        s = all_stats_dict[t]
        dyn_global_stats["total_trades"] += s["total_trades"]
        dyn_global_stats["winning_trades"] += s["winning_trades"]
        dyn_global_stats["losing_trades"] += s["losing_trades"]
        dyn_global_stats["gross_profit"] += s["gross_profit"]
        dyn_global_stats["gross_loss"] += s["gross_loss"]
        dyn_global_stats["trade_durations"].extend(s["trade_durations"])
        if s["max_loss"] < dyn_global_stats["max_loss"]:
            dyn_global_stats["max_loss"] = s["max_loss"]

    fig_global = go.Figure()
    colors = {"Stocks": "#3498db", "Indices": "#e67e22", "Crypto": "#9b59b6"}

    for market_name in selected_categories:
        df_market = market_portfolio_values[market_name]
        if not df_market.empty:
            category_avg = df_market.mean(axis=1)
            fig_global.add_trace(go.Scatter(
                x=category_avg.index, y=category_avg, mode='lines',
                name=f'{market_name} Average',
                line=dict(color=colors.get(market_name, '#bdc3c7'), width=2, dash='dash')
            ))

    fig_global.add_trace(go.Scatter(
        x=avg_portfolio.index, y=avg_portfolio, mode='lines',
        name='Overall Portfolio Average',
        line=dict(color='#27ae60', width=3.5)
    ))

    global_chart_title = "1. Portfolio Performance by Category and Overall Average ($)"
    if APPLY_INFLATION:
        global_chart_title += " [INFLATION ADJUSTED]"

    fig_global.update_layout(title=global_chart_title, xaxis_title="Date", yaxis_title="Average Capital ($)",
                             template="plotly_dark", height=400)
    st.plotly_chart(fig_global, width='stretch')

    fig_winrate = go.Figure()
    fig_winrate.add_trace(go.Scatter(x=avg_win_rate.index, y=avg_win_rate, mode='lines', name='Average Win Rate',
                                     line=dict(color='#f1c40f', width=2.5)))

    first_valid_idx = avg_win_rate.first_valid_index()
    if first_valid_idx is not None:
        first_val = avg_win_rate.loc[first_valid_idx]
        fig_winrate.add_annotation(
            x=first_valid_idx, y=first_val, text="🚩 1st Trade",
            showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2, arrowcolor="#f1c40f",
            ax=0, ay=-40, font=dict(size=12, color="white"),
            bgcolor="#2c3e50", bordercolor="#f1c40f", borderwidth=1.5, borderpad=4
        )

    fig_winrate.add_hline(y=50, line_dash="dash", line_color="white", annotation_text="50% Threshold")
    fig_winrate.update_layout(title="2. Cumulative Win Rate Evolution (%)", xaxis_title="Date",
                              yaxis_title="Win Rate (%)", template="plotly_dark", height=350)
    st.plotly_chart(fig_winrate, width='stretch')

    cum_max = avg_portfolio.cummax()
    drawdown = (avg_portfolio - cum_max) / cum_max
    max_dd = drawdown.min() * 100

    daily_returns = avg_portfolio.pct_change().dropna()

    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0

    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
    sortino_ratio = (daily_returns.mean() / downside_std) * np.sqrt(252) if downside_std != 0 else 0

    total_closed = dyn_global_stats["total_trades"]
    global_win_rate = (dyn_global_stats["winning_trades"] / total_closed * 100) if total_closed > 0 else 0.0

    if total_closed == 0:
        profit_factor_display = "N/A"
        risk_reward_display = "N/A"
        expectancy_display = "N/A"
    else:
        pf = dyn_global_stats["gross_profit"] / dyn_global_stats["gross_loss"] if dyn_global_stats[
                                                                                      "gross_loss"] > 0 else float(
            'inf')
        profit_factor_display = f"{pf:.2f}" if pf != float('inf') else "∞ (No Losses)"

        avg_win = dyn_global_stats["gross_profit"] / dyn_global_stats["winning_trades"] if dyn_global_stats[
                                                                                               "winning_trades"] > 0 else 0
        avg_loss = dyn_global_stats["gross_loss"] / dyn_global_stats["losing_trades"] if dyn_global_stats[
                                                                                             "losing_trades"] > 0 else 0

        rr = avg_win / avg_loss if avg_loss > 0 else float('inf')
        risk_reward_display = f"{rr:.2f}" if rr != float('inf') else "∞ (No Losses)"

        win_prob = dyn_global_stats["winning_trades"] / total_closed
        loss_prob = dyn_global_stats["losing_trades"] / total_closed
        expectancy = (win_prob * avg_win) - (loss_prob * avg_loss)
        expectancy_display = f"{expectancy:.2f} $"

    net_profit = avg_portfolio.iloc[-1] - INITIAL_CAPITAL
    max_dd_amount = abs((INITIAL_CAPITAL * (max_dd / 100)))
    recovery_factor = (net_profit / max_dd_amount) if max_dd_amount > 0 else 0.0

    avg_time_in_market = np.mean(dyn_global_stats["trade_durations"]) if len(
        dyn_global_stats["trade_durations"]) > 0 else 0

    with st.container(border=True):
        box_title = "📊 Dynamic Post-Backtest Statistics (Real Values)" if APPLY_INFLATION else "📊 Dynamic Post-Backtest Statistics (Nominal Values)"
        st.subheader(box_title)

        col_a, col_b, col_c, col_d = st.columns(4)

        col_a.metric("Total Trades", dyn_global_stats["total_trades"])
        col_a.metric("Global Win Rate", f"{global_win_rate:.1f} %")

        col_b.metric("Risk / Reward", risk_reward_display)
        col_b.metric("Profit Factor", profit_factor_display)

        col_c.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        col_c.metric("Sortino Ratio", f"{sortino_ratio:.2f}")

        col_d.metric("Expectancy / Trade", expectancy_display)
        col_d.metric("Recovery Factor", f"{recovery_factor:.2f}")

        col_e, col_f, col_g, col_h = st.columns(4)
        col_e.metric("Max Drawdown (DD)", f"{max_dd:.2f} %")
        col_f.metric("Max Loss", f"{dyn_global_stats['max_loss']:.2f} $")
        col_g.metric("Time in Market (Avg)", f"{avg_time_in_market:.1f} days")
        col_h.metric("Net Profit", f"{net_profit:.2f} $")
