"""
This is the full implementation of the portfolio optimization using AMSP method.
"""
import os
import numpy as np
import pandas as pd
import quantstats as qs
import matplotlib
from qfolio.analysis.comparison import *
from qfolio.analysis.metrics import calculate_performance_stats, key_metrics, calculate_advanced_metrics
from qfolio.backtesting.PortfolioManager_AMSP import PortfolioManager
from qfolio.backtesting.portfolio import calculate_sector_concentration, calculate_sector_exposure
from qfolio.backtesting.date_loader import *
from qfolio.data.DataManager import DataManager
from qfolio.data.assets_loader import get_eligible_stocks
from qfolio.optimization.core import BackTrackCore, OptimizationConfig
from qfolio.screeners.sharpe_screener import *
from qfolio.utils.constants import *
from qfolio.visualization.plotter import *
from qfolio.results.exporter import save_backtest_results

# =============================================================================
# show plots or not, prefer False for grid research
show_plots = True
if not show_plots:
    matplotlib.use('Agg')  # Non-interactive backend for unattended grid search
# =============================================================================

# 1. --- Configuration ---
data = load_data(path="examples/example_csv/SP500_42stocks_baseline_adjusted_close_103025.csv")

# Ensure all data columns are numeric, coercing errors
for col in data.columns: 
    data[col] = pd.to_numeric(data[col], errors='coerce')

sim_start_date = "2020-07-01" # Q3 2020 start
sim_end_date = "2025-10-28" # End on October 28, 2025

rebalance_freq = '63B'
training_lookback_days = 150 # check 60, 90, 120, 150, 180
risk_lookback_period = 120 # Lookback for CVaR calculation (needed for data filtering)
exchange = 'NYSE'

initial_budget = 20000
new_invest_per_period = 0
benchmark_assets = ['SPY']

# Full potential universe (all stocks except benchmark)
trading_universe_full = list(data.columns.drop(benchmark_assets))  # All 51 stocks

print(f"All available assets: {data.columns.tolist()}")
print(f"Total assets in data: {len(data.columns)}")
print(f"Potential trading universe: {len(trading_universe_full)} stocks (excluding {benchmark_assets} benchmark)")

# Keep all data (no pre-filtering - dynamic filtering happens at each rebalance)
all_needed_assets = trading_universe_full + benchmark_assets
data = data[all_needed_assets]

# Dynamic universe: will be updated at each rebalance date based on data availability 
print(f"\nUsing DYNAMIC UNIVERSE approach:")
print(f"  - Stocks become eligible when they have {training_lookback_days + risk_lookback_period} days of history")
print(f"  - Universe checked at each rebalance (every {rebalance_freq})")
print(f"  - Follows MSCI index methodology for new stock inclusion\n")

# --- Optimization Parameters ---
k = 2
lambda1 = 1E12 # E12 for classic/ QAOA. E3 - E6 can have more stable results ~ 35, 36%
#TODO Test lambda1 in [1E6, 1E9, 1E12], q in [1E-4, 1E-3, 1E-2]
q = 1e-2
H_scale = 1E5 # testing E5 for QAOA/ classic, 1 ffor SCIP and exact_miqp
solver_type = 'classic' # classic, SamplerVQE, QAOA, QAOA_MPS, QAOA_SPSA, SCIP, exact_miqp, QAOA_shots

# --- Risk Manager Parameters ---
sharpe_n = 3
confidence_level = 0.95

# 2. --- Setup with Market Calendar ---
all_trading_days = get_all_trading_date(exchange, 
                                        sim_start_date, 
                                        sim_end_date, 
                                        training_lookback_days, 
                                        risk_lookback_period)
# Get rebalance date 
actual_rebalance_dates = get_rebalance_dates(start_date=sim_start_date, end_date=sim_end_date,data=data, freq=rebalance_freq)

# Get initial eligible universe
initial_train_start, initial_train_end = initial_eligible_universe(all_trading_days,actual_rebalance_dates,training_lookback_days)

# Determine initial eligible stocks based on data availability
initial_eligible_stocks = get_eligible_stocks(
    current_date=initial_train_end,
    all_stocks=trading_universe_full,
    data=data,
    lookback_days=training_lookback_days + risk_lookback_period
)

print(f"Initial eligible universe (as of {initial_train_end.strftime('%Y-%m-%d')}): {len(initial_eligible_stocks)} stocks")
print(f"  Stocks: {initial_eligible_stocks}\n")

# Calculate initial Sharpe ratios on eligible stocks only
initial_assets = get_initial_assets(data, 
                                    initial_eligible_stocks, 
                                    initial_train_start, 
                                    initial_train_end, 
                                    sharpe_n)

# Initialize PortfolioManager with initial eligible universe
PM = PortfolioManager(data=data[initial_eligible_stocks],
                    budget=initial_budget,
                    new_invest=new_invest_per_period,
                    assets_portfolio=initial_assets,
                    assets_benchmark=benchmark_assets,
                    screener=SharpeRatioCalculator)

# --- New Tracking Variables for Enhanced Analytics ---
composition_history = []  # Track which stocks selected at each rebalance
turnover_history = []  # Track portfolio turnover at each rebalance
previous_allocation = {}  # For turnover calculation
sector_exposure_history = []  # Track sector weights at each rebalance
concentration_history = []  # Track sector concentration (HHI) over time

simulation_days = all_trading_days[all_trading_days.slice_indexer(sim_start_date, sim_end_date)]

# Create optimization configuration
opt_config = OptimizationConfig(
    k=k,
    lambda1=lambda1,
    q=q,
    H_scale=H_scale,
    solver_type=solver_type
)

print(f"\n--- Starting Step-by-Step Simulation Using '{exchange}' Calendar ---")
Run_backtrack = BackTrackCore(
    data=data,
    budget=initial_budget,
    sp_n=sharpe_n,
    actual_rebalance_dates=actual_rebalance_dates,
    simulation_days=simulation_days,
    risk_lookback_period=risk_lookback_period,
    initial_eligible_stocks=initial_eligible_stocks,
    PortfolioManager=PM,
    new_invest_per_period=new_invest_per_period,
    all_trading_days=all_trading_days,
    training_lookback_days = training_lookback_days,
    trading_universe_full=trading_universe_full,
    opt_config=opt_config)

portfolio_history, all_rebalance_events = Run_backtrack.run()

port_roi, port_values, bm_roi, bm_values, bm = comparison(portfolio_history,
                                                      data = data,
                                                      sim_start_date = sim_start_date,
                                                      sim_end_date = sim_end_date,
                                                      initial_budget=initial_budget,
                                                      new_invest_per_period=new_invest_per_period,
                                                      actual_rebalance_dates=actual_rebalance_dates)

# Portfolio Statistics
port_stats = calculate_performance_stats(portfolio_history, 
                                         port_values,
                                        port_roi)

# --- Upside/Downside Capture Ratios ---
opt_daily_returns = port_stats['Daily Return (%)']
spy_daily_returns = bm_values['SPY'].pct_change().dropna()
upside_capture, downside_capture, capture_ratio = up_downside_captrue_ratio(opt_daily_returns, spy_daily_returns)

# --- Drawdown Duration Analysis ---
opt_drawdown = port_stats['Drawdown (%)']
max_drawdown_days, avg_drawdown_days, spy_max_drawdown_days, spy_avg_drawdown_days = drawdown_duration(opt_drawdown, spy_daily_returns)

# Print all key metrics
key_metrics(portfolio_history, port_values, port_roi, bm_values, bm_roi, bm)
# --- Plotting ROI ---
plot_roi_comp(port_roi, bm_roi, all_rebalance_events, exchange)
# print portfolio values
print(port_values)
# --- Daily Value Comparison Chart ---
plot_dv_comparison(port_values, bm_values, initial_budget, exchange)
# --- Original Portfolio Value Chart (Keep for reference) ---
plot_port_val(port_values, exchange)

# --- Portfolio Composition & Turnover Over Time ---
# NOTE: These plots require advanced=True in BackTrackCore.run()
if composition_history and turnover_history:
    plot_composition_n_turnover(composition_history, turnover_history, sharpe_n)
# --- Sector Exposure Stacked Area Chart ---
if sector_exposure_history:
    plot_sector_exposure(sector_exposure_history)
# --- Sector Exposure Heatmap ---
if sector_exposure_history:
    plot_se_history(sector_exposure_history)
# --- Sector Concentration (HHI) Over Time ---
if concentration_history:
    plot_conecntration(concentration_history)

# ===============================================================================
# QUANTSTATS HTML REPORT GENERATION
# ===============================================================================

print("\n" + "="*90)
print("Generating QuantStats HTML Report...")
print("="*90)

# Prepare returns for QuantStats (needs pandas Series with DatetimeIndex)
opt_returns_for_qs = opt_daily_returns.copy()
spy_returns_for_qs = spy_daily_returns.copy()

# Ensure both have same index
common_index = opt_returns_for_qs.index.intersection(spy_returns_for_qs.index)
opt_returns_for_qs = opt_returns_for_qs.loc[common_index]
spy_returns_for_qs = spy_returns_for_qs.loc[common_index]

# Create output directory
output_dir = "quantstats_reports"
os.makedirs(output_dir, exist_ok=True)

# Generate comprehensive HTML report for optimized portfolio
report_filename = f"{output_dir}/portfolio_report_{sim_start_date}_{sim_end_date}.html"
print(f"\nGenerating full tearsheet: {report_filename}")

try:
    qs.reports.html(
        opt_returns_for_qs,
        benchmark=spy_returns_for_qs,
        output=report_filename,
        title=f"Quantum Portfolio Optimization - Full Tearsheet ({sim_start_date} to {sim_end_date})",
        download_filename=report_filename
    )
    print(f"[OK] Full HTML report saved: {report_filename}")
except Exception as e:
    print(f"[ERROR] Error generating HTML report: {e}")

print("\n" + "="*90)
print("QuantStats HTML report generation completed!")
print(f"Output directory: {os.path.abspath(output_dir)}")
print("="*90)

# ===============================================================================
# SAVE RESULTS TO JSON (for grid search analysis)
# ===============================================================================

# Save results
# Calculate advanced metrics once for export
advanced = calculate_advanced_metrics(
    opt_daily_returns=opt_daily_returns,
    spy_daily_returns=spy_daily_returns,
    opt_drawdown=opt_drawdown
)

save_backtest_results(
    portfolio_history=portfolio_history,
    port_stats=port_stats,
    spy_stats=bm.get_statistics(bm_values, bm_roi)['SPY'],
    opt_config=opt_config,
    run_config={
        'sim_start_date': sim_start_date,
        'sim_end_date': sim_end_date,
        'initial_budget': initial_budget,
        'new_invest_per_period': new_invest_per_period,
        'training_lookback_days': training_lookback_days,
        'rebalance_freq': rebalance_freq,
        'exchange': exchange,
        'sharpe_n': sharpe_n,
        'confidence_level': confidence_level
    },
    output_dir='results',
    include_timeseries=False,
    advanced_metrics=advanced  # Include all QuantStats metrics for grid search
)

print("\n" + "="*90)
print("Results saved for grid search analysis")
print("="*90 + "\n")

