"""
Backtest of magnificent 7 stocks 
"""

import sys
import numpy as np
import os
import pandas as pd
from pandas.tseries.offsets import BDay
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal
from qfolio.data.DataManager import DataManager
from qfolio.backtesting.AMSP_PortfolioManager_V2 import PortfolioManager
from data_screener.screener import *
from qfolio.metrics.RiskMetrics import RiskMetricsCalculator 

# 1. --- Configuration ---
data = load_data(path="examples/example_csv/SP500_42stocks_baseline_adjusted_close_103025.csv")

# Ensure all data columns are numeric, coercing errors
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

trading_universe  = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA']#, 'SH']

print(f"Assets in universe: {data.columns.tolist()}")
print(f"Total assets: {len(data.columns)}")

sim_start_date = "2023-01-01"
sim_end_date = "2025-09-01"

rebalance_freq = '63B'
training_lookback_days = 120 # check 60, 90, 120, 150, 180
exchange = 'NYSE'

initial_budget = 10000
new_invest_per_period = 0
benchmark_assets = ['VOO']

all_needed_assets = trading_universe + benchmark_assets
data = data[all_needed_assets]

data_for_optimization = data[trading_universe]

# --- Optimization Parameters ---
k = 2
lambda1 = 1E12 
q = 1E-3
H_scale = 1E5 
solver_type = 'classic' # classic, QAOA, QAOA_shots, SamplerVQE

# --- Risk Manager Parameters ---
sharpe_n = 3
confidence_level = 0.95
risk_lookback_period = 120 # Lookback for CVaR calculation

# 2. --- Setup with Market Calendar ---
market_cal = mcal.get_calendar(exchange)
data_start_date_for_calendar = (pd.to_datetime(sim_start_date) - BDay(training_lookback_days + risk_lookback_period + 5)).strftime('%Y-%m-%d')
full_schedule = market_cal.schedule(start_date=data_start_date_for_calendar, end_date=sim_end_date)
all_trading_days = full_schedule.index

approx_rebalance_dates = pd.date_range(start=sim_start_date, end=sim_end_date, freq=rebalance_freq)
rebalance_indices = all_trading_days.searchsorted(approx_rebalance_dates, side='left')
rebalance_indices = rebalance_indices[rebalance_indices < len(all_trading_days)]
actual_rebalance_dates = all_trading_days[rebalance_indices]

initial_train_end_loc = all_trading_days.get_loc(actual_rebalance_dates[0]) - 1
initial_train_end = all_trading_days[initial_train_end_loc]
initial_train_start_loc = initial_train_end_loc - training_lookback_days
if initial_train_start_loc < 0: initial_train_start_loc = 0
initial_train_start = all_trading_days[initial_train_start_loc]
initial_sharpe_results = SharpeRatioCalculator(data, initial_train_start.strftime('%Y-%m-%d'),
initial_train_end.strftime('%Y-%m-%d'), risk_free=0.0, print_out=False)

initial_positive_mean_returns = initial_sharpe_results['r_i_series'][initial_sharpe_results['r_i_series'] > 0]
if not initial_positive_mean_returns.empty:
    initial_candidate_sharpes = initial_sharpe_results['sharpe_series'][initial_positive_mean_returns.index]
    initial_assets = initial_candidate_sharpes.dropna().sort_values(ascending=False).head(sharpe_n).index.tolist()
else:
    initial_assets = []

PM = PortfolioManager(data=data_for_optimization,
                    budget=initial_budget,
                    new_invest=new_invest_per_period,
                    assets_portfolio=initial_assets,
                    assets_benchmark=benchmark_assets,
                    screener=SharpeRatioCalculator)

# 3. --- Main Simulation Loop (Daily) ---
portfolio_history = []
current_budget = initial_budget
last_allocation = {}
portfolio_high_value = initial_budget
all_rebalance_events = []

simulation_days = all_trading_days[all_trading_days.slice_indexer(sim_start_date, sim_end_date)]

print(f"\n--- Starting Step-by-Step Simulation Using '{{exchange}}' Calendar ---")
for i, current_date in enumerate(simulation_days):
    #print(f"\n{'='*20} Day {i+1}: {current_date.strftime('%Y-%m-%d')} {'='*20}")

    # Calculate portfolio value for the current day
    if last_allocation and (current_date in data.index):
        prices_today = data.loc[current_date].to_dict()
        current_portfolio_value = sum(last_allocation.get(ticker, 0) * prices_today.get(ticker, 0) for ticker in last_allocation)
    else:
        current_portfolio_value = current_budget

    portfolio_high_value = max(portfolio_high_value, current_portfolio_value)
    portfolio_history.append({'date': current_date, 'value': current_portfolio_value, 'allocation': last_allocation})

    # --- CVaR Risk Monitoring (For Analysis Only - Does Not Trigger Trades) ---
    if last_allocation:
        try:
            risk_calc = RiskMetricsCalculator(
                portfolio_allocation=last_allocation,
                price_data=data,
                current_date=current_date,
                lookback_period=risk_lookback_period
            )
            portfolio_risk = risk_calc.calculate_portfolio_metrics(confidence_level=confidence_level, method='historical')
            cvar_95 = portfolio_risk['CVaR']
            var_95 = portfolio_risk['VaR']

            # MONITORING ONLY - print if risk is elevated, but don't act on it
            if cvar_95 < -0.10:  # Only warn if risk is significant
                print(f"  [RISK MONITOR] {current_date.strftime('%Y-%m-%d')}: CVaR = {cvar_95:.2%}, VaR = {var_95:.2%}")
        except Exception as e:
            pass  # Silent failure for monitoring

    # --- Rebalancing Logic (Only on Scheduled Dates) ---
    is_regular_rebalance = current_date in actual_rebalance_dates

    if is_regular_rebalance:
        all_rebalance_events.append(current_date)
        current_budget_for_opt = current_portfolio_value
        current_budget_for_opt += new_invest_per_period

        print(f"--- Rebalancing on {current_date.strftime('%Y-%m-%d')} --- (Budget for opt: ${current_budget_for_opt:.2f})")

        current_date_loc = all_trading_days.get_loc(current_date)
        train_end_loc = current_date_loc - 1
        if train_end_loc < 0:
            print("Not enough historical data to train. Skipping rebalance.")
            continue
        train_end = all_trading_days[train_end_loc]
        train_start_loc = train_end_loc - training_lookback_days
        if train_start_loc < 0: train_start_loc = 0
        train_start = all_trading_days[train_start_loc]

        current_sharpe_results = PM.monitor_and_select_assets(train_start.strftime('%Y-%m-%d'), train_end.strftime('%Y-%m-%d'),
                                                               top_n=sharpe_n)

        if not PM.assets_portfolio:
            print("--- No assets with positive Sharpe. Maintaining previous allocation. ---")
            if last_allocation:
                # Keep previous allocation, just update value with current prices
                current_portfolio_value = sum(
                    last_allocation.get(ticker, 0) * data.loc[current_date, ticker]
                    for ticker in last_allocation if ticker in data.columns
                )
                portfolio_history[-1]['value'] = current_portfolio_value
            continue

        selected_assets_r_i = current_sharpe_results["r_i_series"].loc[PM.assets_portfolio]
        if all(selected_assets_r_i < 0):
            print("--- All selected assets have negative expected returns. Maintaining previous allocation. ---")
            if last_allocation:
                # Keep previous allocation, just update value with current prices
                current_portfolio_value = sum(
                    last_allocation.get(ticker, 0) * data.loc[current_date, ticker]
                    for ticker in last_allocation if ticker in data.columns
                )
                portfolio_history[-1]['value'] = current_portfolio_value
            continue

        opt_result = PM.run_single_optimization(
            current_date=current_date.strftime('%Y-%m-%d'),
            train_start=train_start.strftime('%Y-%m-%d'),
            train_end=train_end.strftime('%Y-%m-%d'),
            budget=current_budget_for_opt,
            k=k, lambda1=lambda1, q=q, H_scale=H_scale, solver_type=solver_type,
            latest_open_prices=data.loc[current_date].to_dict()
        )
        last_allocation = opt_result['allocation']
        portfolio_history[-1]['allocation'] = last_allocation
        portfolio_history[-1]['value'] = opt_result['value']
        portfolio_high_value = opt_result['value']
        print(f"End of Rebalance Period. Portfolio Value: ${opt_result['value']:.2f}")

print("--- Simulation Complete ---")

# 4. --- Results and Plotting ---
if not portfolio_history:
    print("Simulation did not produce any results.")
    sys.exit()

# --- Process Optimized Portfolio Results ---
opt_res_df = pd.DataFrame(portfolio_history)
opt_res_df['date'] = pd.to_datetime(opt_res_df['date'])
opt_res_df = opt_res_df.set_index('date')['value']

# Calculate ROI for Optimized Portfolio
invested_series_opt = pd.Series(0.0, index=opt_res_df.index)
if not opt_res_df.empty:
    # Initial investment
    invested_series_opt.iloc[0] = initial_budget

    # Subsequent investments on scheduled rebalance dates only
    for date in actual_rebalance_dates:
        if date in invested_series_opt.index:
            invested_series_opt[date] += new_invest_per_period
    
    invested_series_opt = invested_series_opt.cumsum()

roi_optimized = ((opt_res_df - invested_series_opt) / invested_series_opt) * 100

# --- Process VOO Benchmark Results (NEW: Daily Tracking) ---
from qfolio.backtesting.DailyBenchmarkSimulator import DailyBenchmarkSimulator
voo_benchmark = DailyBenchmarkSimulator(
    assets=['VOO'],
    data=data,
    init_budget=initial_budget,
    new_investment=new_invest_per_period
)

# Run daily simulation with same rebalance dates as optimized portfolio
voo_daily_values, voo_daily_roi = voo_benchmark.simulate(
    start_date=sim_start_date,
    end_date=sim_end_date,
    rebalance_dates=actual_rebalance_dates  # Same dates as optimized portfolio!
)

# Extract VOO series for compatibility
voo_bm_results = voo_daily_roi['VOO']
voo_values = voo_daily_values['VOO']

# --- Calculate Performance Statistics ---
# Optimized Portfolio Statistics
opt_daily_returns = opt_res_df.pct_change().dropna()
opt_total_return = roi_optimized.iloc[-1]
opt_volatility = opt_daily_returns.std() * np.sqrt(252) * 100
opt_sharpe = (opt_daily_returns.mean() / opt_daily_returns.std()) * np.sqrt(252) if opt_daily_returns.std() > 0 else 0

# Calculate max drawdown for optimized portfolio
opt_cumulative = (1 + opt_daily_returns).cumprod()
opt_running_max = opt_cumulative.cummax()
opt_drawdown = (opt_cumulative - opt_running_max) / opt_running_max
opt_max_drawdown = opt_drawdown.min() * 100

# Win rate for optimized portfolio
opt_win_rate = (opt_daily_returns > 0).sum() / len(opt_daily_returns) * 100 if len(opt_daily_returns) > 0 else 0

# Days in market (not in cash)
days_in_market = sum(1 for record in portfolio_history if len(record['allocation']) > 0)
total_days = len(portfolio_history)

# VOO Statistics (using built-in method)
voo_stats = voo_benchmark.get_statistics(voo_daily_values, voo_daily_roi)['VOO']

# --- Print Comprehensive Comparison ---
print("\n" + "="*70)
print(" " * 20 + "PERFORMANCE COMPARISON")
print("="*70)
print(f"\n{'Metric':<30} {'Optimized':<20} {'VOO Benchmark':<20}")
print("-"*70)
print(f"{'Total Return (%)':<30} {opt_total_return:>18.2f}% {voo_stats['Total Return (%)']:>18.2f}%")
print(f"{'Volatility (annual %)':<30} {opt_volatility:>18.2f}% {voo_stats['Volatility (%)']:>18.2f}%")
print(f"{'Sharpe Ratio':<30} {opt_sharpe:>18.2f}  {voo_stats['Sharpe Ratio']:>18.2f}")
print(f"{'Max Drawdown (%)':<30} {opt_max_drawdown:>18.2f}% {voo_stats['Max Drawdown (%)']:>18.2f}%")
print(f"{'Win Rate (%)':<30} {opt_win_rate:>18.2f}% {voo_stats['Win Rate (%)']:>18.2f}%")
print(f"{'Days in Market':<30} {days_in_market:>18} / {total_days:<16}")
print(f"{'Total Trading Days':<30} {total_days:>18}  {voo_stats['Total Days']:>18}")
print("-"*70)

# Performance delta
roi_diff = opt_total_return - voo_stats['Total Return (%)']
sharpe_diff = opt_sharpe - voo_stats['Sharpe Ratio']
print(f"\n{'Performance vs VOO:':<30}")
print(f"{'  Return Difference':<30} {roi_diff:>18.2f}%  {'(Better)' if roi_diff > 0 else '(Worse)'}")
print(f"{'  Sharpe Difference':<30} {sharpe_diff:>18.2f}   {'(Better)' if sharpe_diff > 0 else '(Worse)'}")
print("="*70 + "\n")

# --- Plotting ROI ---
plt.figure(figsize=(12, 6))
plt.plot(roi_optimized, label="Optimized Portfolio ROI", linewidth=2, color='black')
plt.plot(voo_bm_results, label="100% VOO Benchmark ROI", linewidth=2, linestyle='--', color='blue')

# Highlight ALL rebalance events
if all_rebalance_events:
    for i, date in enumerate(all_rebalance_events):
        plt.axvline(x=date, color='red', linestyle='--', linewidth=1, label='Rebalance Date' if i == 0 else "")

plt.title(f"Portfolio ROI vs. 100% VOO Benchmark ({exchange})")
plt.xlabel("Date")
plt.ylabel("Return on Investment (%)")
plt.axhline(0, color='grey', linestyle='--') # Add a zero line for reference
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(opt_res_df)

# --- NEW: Daily Value Comparison Chart ---
plt.figure(figsize=(14, 7))
plt.plot(opt_res_df, label="Optimized Portfolio", linewidth=2.5, color='black')
plt.plot(voo_values, label="VOO Buy-and-Hold", linewidth=2.5, linestyle='--', color='blue')

# Add initial investment reference line
plt.axhline(y=initial_budget, color='gray', linestyle=':', alpha=0.5, linewidth=1, label=f'Initial: ${initial_budget}')

# Highlight rebalance dates with vertical lines
for i, date in enumerate(actual_rebalance_dates):
    if date in opt_res_df.index:
        plt.axvline(x=date, color='red', alpha=0.2, linestyle='--', linewidth=0.8,
                   label='Rebalance Date' if i == 0 else "")

# Shade regions where optimized beats VOO
opt_beats_voo = opt_res_df > voo_values
if opt_beats_voo.any():
    plt.fill_between(opt_res_df.index, opt_res_df, voo_values,
                     where=opt_beats_voo, alpha=0.2, color='green',
                     label='Optimized Outperforms', interpolate=True)
    plt.fill_between(opt_res_df.index, opt_res_df, voo_values,
                     where=~opt_beats_voo, alpha=0.2, color='red',
                     label='VOO Outperforms', interpolate=True)

plt.title(f"Portfolio Value Comparison: Optimized vs VOO ({exchange})\n" +
          f"Training Window: {training_lookback_days} days | Rebalance: Every {rebalance_freq}",
          fontsize=14, fontweight='bold')
plt.ylabel("Portfolio Value ($)", fontsize=12)
plt.xlabel("Date", fontsize=12)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xticks(rotation=45)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- Original Portfolio Value Chart (Keep for reference) ---
plt.figure(figsize=(12, 6))
plt.plot(opt_res_df, label="Optimized Portfolio Value", linewidth=2, color='black')
plt.title(f"Portfolio Value Over Time ({exchange})")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()