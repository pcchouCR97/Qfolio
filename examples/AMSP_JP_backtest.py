"""
JPMorgan Quantum Portfolio Backtest 

This script uses the EXACT SAME backtesting framework as AMSP_example.py
but replaces the standard optimization with JPMorgan decomposition pipeline.

Features:
1. Same date handling and calendar logic
2. Same dynamic universe approach
3. Same Sharpe screening (pick top N assets)
4. JPMorgan pipeline: RMT + Spectral Clustering + QAOA
5. Same performance metrics and visualization
"""

import os
import numpy as np
import pandas as pd
import quantstats as qs
import matplotlib
from qfolio.analysis.comparison import *
from qfolio.analysis.metrics import calculate_performance_stats, key_metrics, calculate_advanced_metrics
from qfolio.backtesting.date_loader import *
from qfolio.data.assets_loader import get_eligible_stocks
from qfolio.screeners.sharpe_screener import *
from qfolio.utils.constants import *
from qfolio.visualization.plotter import *
from qfolio.results.exporter import save_backtest_results

# JPMorgan-specific imports
from qfolio.optimization.jpmorgan_pipeline import run_jpmorgan_pipeline

# =============================================================================
# Configuration
# =============================================================================
show_plots = True
if not show_plots:
    matplotlib.use('Agg')

# 1. --- Load Data ---
data = load_data(path="examples/example_csv/SP500_42stocks_baseline_adjusted_close_103025.csv")

for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

sim_start_date = "2020-07-01"
sim_end_date = "2025-10-28"

rebalance_freq = '63B'
training_lookback_days = 150  # Increased for JPMorgan (needs more data)
risk_lookback_period = 120
exchange = 'NYSE'

initial_budget = 20000
new_invest_per_period = 0
benchmark_assets = ['SPY']

trading_universe_full = list(data.columns.drop(benchmark_assets))

print(f"All available assets: {data.columns.tolist()}")
print(f"Total assets in data: {len(data.columns)}")
print(f"Potential trading universe: {len(trading_universe_full)} stocks")

all_needed_assets = trading_universe_full + benchmark_assets
data = data[all_needed_assets]

print(f"\nUsing DYNAMIC UNIVERSE approach:")
print(f"  - Stocks become eligible when they have {training_lookback_days + risk_lookback_period} days of history")
print(f"  - Universe checked at each rebalance (every {rebalance_freq})")

# --- JPMorgan Optimization Parameters ---
sharpe_n = 9  # Pre-filter top 9 assets by Sharpe
max_community_size = 3
max_binary_bits = 7
k = 2
q = 1e-3
lambda1 = 1E12
H_scale = 1E5

print(f"\nJPMorgan Pipeline Configuration:")
print(f"  Top N assets (Sharpe screening): {sharpe_n}")
print(f"  Max community size: {max_community_size}")
print(f"  Risk parameter q: {q}")
print(f"  Lambda1 (budget penalty): {lambda1}")

# 2. --- Setup with Market Calendar ---
all_trading_days = get_all_trading_date(exchange, sim_start_date, sim_end_date,
                                        training_lookback_days, risk_lookback_period)

actual_rebalance_dates = get_rebalance_dates(start_date=sim_start_date,
                                             end_date=sim_end_date,
                                             data=data,
                                             freq=rebalance_freq)

initial_train_start, initial_train_end = initial_eligible_universe(all_trading_days,
                                                                   actual_rebalance_dates,
                                                                   training_lookback_days)

initial_eligible_stocks = get_eligible_stocks(
    current_date=initial_train_end,
    all_stocks=trading_universe_full,
    data=data,
    lookback_days=training_lookback_days + risk_lookback_period
)

print(f"\nInitial eligible universe (as of {initial_train_end.strftime('%Y-%m-%d')}): {len(initial_eligible_stocks)} stocks")
print(f"  Stocks: {initial_eligible_stocks}\n")

# Get initial top N assets
initial_assets = get_initial_assets(data, initial_eligible_stocks,
                                   initial_train_start, initial_train_end, sharpe_n)

print(f"Initial top {sharpe_n} assets: {initial_assets}\n")

# 3. --- Main Simulation Loop ---
portfolio_history = []
current_budget = initial_budget
last_allocation = {}
portfolio_high_value = initial_budget
all_rebalance_events = []
is_first_rebalance = True
previous_eligible_stocks = initial_eligible_stocks

simulation_days = all_trading_days[all_trading_days.slice_indexer(sim_start_date, sim_end_date)]

print(f"\n--- Starting JPMorgan Pipeline Backtest Using '{exchange}' Calendar ---")

for i, current_date in enumerate(simulation_days):
    # Calculate portfolio value for the current day
    if last_allocation and (current_date in data.index):
        prices_today = data.loc[current_date].to_dict()
        current_portfolio_value = sum(
            last_allocation.get(ticker, 0) * prices_today.get(ticker, 0)
            for ticker in last_allocation
        )
    else:
        current_portfolio_value = current_budget

    portfolio_high_value = max(portfolio_high_value, current_portfolio_value)
    portfolio_history.append({
        'date': current_date,
        'value': current_portfolio_value,
        'allocation': last_allocation
    })

    # --- Rebalancing Logic (Only on Scheduled Dates) ---
    is_regular_rebalance = current_date in actual_rebalance_dates

    if is_regular_rebalance:
        all_rebalance_events.append(current_date)
        current_budget_for_opt = current_portfolio_value

        if not is_first_rebalance:
            current_budget_for_opt += new_invest_per_period
            print(f"\n--- Rebalancing on {current_date.strftime('%Y-%m-%d')} --- (Budget: ${current_portfolio_value:.2f} + New: ${new_invest_per_period:.2f} = ${current_budget_for_opt:.2f})")
        else:
            print(f"\n--- First Rebalancing on {current_date.strftime('%Y-%m-%d')} --- (Budget: ${current_budget_for_opt:.2f})")
            is_first_rebalance = False

        current_date_loc = all_trading_days.get_loc(current_date)
        train_end_loc = current_date_loc - 1
        if train_end_loc < 0:
            print("Not enough historical data. Skipping.")
            continue

        train_end = all_trading_days[train_end_loc]
        train_start_loc = train_end_loc - training_lookback_days
        if train_start_loc < 0:
            train_start_loc = 0
        train_start = all_trading_days[train_start_loc]

        # --- Dynamic Universe Update ---
        current_eligible_stocks = get_eligible_stocks(
            current_date=train_end,
            all_stocks=trading_universe_full,
            data=data,
            lookback_days=training_lookback_days + risk_lookback_period
        )

        if len(current_eligible_stocks) > len(previous_eligible_stocks):
            new_stocks = set(current_eligible_stocks) - set(previous_eligible_stocks)
            print(f"  [UNIVERSE EXPANSION] +{len(new_stocks)} new stocks: {sorted(new_stocks)}")
            previous_eligible_stocks = current_eligible_stocks

        # --- Screen Top N Assets by Sharpe ---
        data_for_screening = data[current_eligible_stocks]
        sharpe_results = SharpeRatioCalculator(data_for_screening,
                                              train_start.strftime('%Y-%m-%d'),
                                              train_end.strftime('%Y-%m-%d'),
                                              risk_free=0.0,
                                              print_out=False)

        positive_mean_returns = sharpe_results['r_i_series'][sharpe_results['r_i_series'] > 0]
        if not positive_mean_returns.empty:
            candidate_sharpes = sharpe_results['sharpe_series'][positive_mean_returns.index]
            selected_assets = candidate_sharpes.dropna().sort_values(ascending=False).head(sharpe_n).index.tolist()
        else:
            selected_assets = []

        print(f"  Screening: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
        print(f"  Top {sharpe_n} assets selected: {selected_assets}")

        if not selected_assets:
            print("  No assets with positive Sharpe. Maintaining previous allocation.")
            continue

        # Check if all selected assets have negative expected returns
        selected_assets_r_i = sharpe_results["r_i_series"].loc[selected_assets]
        if all(selected_assets_r_i < 0):
            print("  All selected assets have negative expected returns. Maintaining previous allocation.")
            continue

        # --- Run JPMorgan Pipeline ---
        print(f"  Running JPMorgan Pipeline on {len(selected_assets)} assets...")

        # Get training data for selected assets
        train_data = data.loc[train_start:train_end, selected_assets]
        current_prices = train_data.iloc[-1].values

        try:
            weights = run_jpmorgan_pipeline(
                prices_data=train_data,
                tickers=selected_assets,
                budget=current_budget_for_opt,
                current_prices=current_prices,
                quantum_solver='classic',
                max_community_size=max_community_size,
                max_binary_bits=max_binary_bits,
                k=k,
                q_original=q,
                lambda1=lambda1
            )

            # Convert weights to shares
            new_allocation = {}
            for idx, ticker in enumerate(selected_assets):
                if weights[idx] > 0:
                    shares = (weights[idx] * current_budget_for_opt) / current_prices[idx]
                    new_allocation[ticker] = shares

            last_allocation = new_allocation
            portfolio_history[-1]['allocation'] = last_allocation

            # Calculate actual value
            actual_value = sum(new_allocation.get(t, 0) * data.loc[current_date, t]
                             for t in new_allocation if t in data.columns)
            portfolio_history[-1]['value'] = actual_value
            portfolio_high_value = actual_value

            # Print allocation details
            print(f"{len(new_allocation)} assets | ${actual_value:,.0f}")
            for ticker in sorted(new_allocation.keys()):
                shares = new_allocation[ticker]
                price = data.loc[current_date, ticker]
                cost = shares * price
                print(f"      {ticker:6s} {shares:6.2f} shares @ ${price:8.2f} = ${cost:10,.2f}")
            
        except Exception as e:
            print(f"  WARNING: JPMorgan optimization failed: {e}")
            print("  Maintaining previous allocation.")
            continue

print("\n--- Simulation Complete ---")

# 4. --- Results and Comparison ---
if not portfolio_history:
    print("Simulation produced no results.")
    exit()

port_roi, port_values, bm_roi, bm_values, bm = comparison(
    portfolio_history,
    data=data,
    sim_start_date=sim_start_date,
    sim_end_date=sim_end_date,
    initial_budget=initial_budget,
    new_invest_per_period=new_invest_per_period,
    actual_rebalance_dates=actual_rebalance_dates
)

# Portfolio Statistics
port_stats = calculate_performance_stats(portfolio_history, port_values, port_roi)

# Upside/Downside Capture
opt_daily_returns = port_stats['Daily Return (%)']
spy_daily_returns = bm_values['SPY'].pct_change().dropna()
upside_capture, downside_capture, capture_ratio = up_downside_captrue_ratio(
    opt_daily_returns, spy_daily_returns
)

# Drawdown Duration
opt_drawdown = port_stats['Drawdown (%)']
max_drawdown_days, avg_drawdown_days, spy_max_drawdown_days, spy_avg_drawdown_days = drawdown_duration(
    opt_drawdown, spy_daily_returns
)

# Print Key Metrics
key_metrics(portfolio_history, port_values, port_roi, bm_values, bm_roi, bm)

# Plotting
plot_roi_comp(port_roi, bm_roi, all_rebalance_events, exchange)
print(port_values)
plot_dv_comparison(port_values, bm_values, initial_budget, exchange)
plot_port_val(port_values, exchange)

# QuantStats Report
print("\n" + "="*90)
print("Generating QuantStats HTML Report...")
print("="*90)

opt_returns_for_qs = opt_daily_returns.copy()
spy_returns_for_qs = spy_daily_returns.copy()

common_index = opt_returns_for_qs.index.intersection(spy_returns_for_qs.index)
opt_returns_for_qs = opt_returns_for_qs.loc[common_index]
spy_returns_for_qs = spy_returns_for_qs.loc[common_index]

output_dir = "quantstats_reports_JP"
os.makedirs(output_dir, exist_ok=True)

report_filename = f"{output_dir}/portfolio_report_JP_{sim_start_date}_{sim_end_date}.html"

try:
    qs.reports.html(
        opt_returns_for_qs,
        benchmark=spy_returns_for_qs,
        output=report_filename,
        title=f"JPMorgan Quantum Portfolio - Full Tearsheet ({sim_start_date} to {sim_end_date})",
        download_filename=report_filename
    )
    print(f"[OK] Full HTML report saved: {report_filename}")
except Exception as e:
    print(f"[ERROR] Error generating HTML report: {e}")

print("\n" + "="*90)
print("Backtest Complete!")
print("="*90 + "\n")
