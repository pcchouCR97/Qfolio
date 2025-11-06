"""
Hybrid Quantum-Classical Portfolio Optimization - Academic Research Version

HYBRID METHODOLOGY:
- Stage 1: QAOA quantum optimization produces coarse discrete solution
- Stage 2: Classical refinement (SLSQP) produces high-precision continuous solution
- Demonstrates quantum-classical collaboration for improved performance

METHODOLOGY:
- Pure systematic strategy with ZERO discretionary decisions
- Portfolio stays fully invested (like benchmark SPY)
- Risk managed through optimization (q parameter), not cash-out
- CVaR calculated for monitoring/analysis only
- Comparable to passive buy-and-hold benchmark

DYNAMIC UNIVERSE METHODOLOGY:
- Stocks enter universe when they have sufficient historical data
- Eligibility checked at each rebalance date
- Follows MSCI index methodology (240-day minimum history)
- Avoids survivorship bias while capturing new opportunities
- No look-ahead bias: only uses data available at decision time

"""

import os
import pandas as pd
from pandas.tseries.offsets import BDay
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for unattended grid search
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal
from qfolio.data.DataManager import DataManager
from qfolio.backtesting.PortfolioManager_AMSP import PortfolioManager
from data_screener.screener import *
from qfolio.metrics.RiskMetrics import RiskMetricsCalculator
from qfolio.optimization.ClassicalRefiner import refine_portfolio 
from qfolio.results.save_results_module import save_backtest_results
import sys
import numpy as np
import seaborn as sns
import quantstats as qs
from scipy import stats

# --- Dynamic Universe Helper Function ---
def get_eligible_stocks(current_date, all_stocks, data, lookback_days=240):
    """
    Returns stocks with sufficient historical data as of current_date.

    Following MSCI index methodology: stocks are eligible for selection once they
    have sufficient trading history for the required lookback window.

    Parameters:
    -----------
    current_date : pd.Timestamp
        The date at which to evaluate eligibility
    all_stocks : list
        Full list of potential stocks to evaluate
    data : pd.DataFrame
        Price data with stocks as columns
    lookback_days : int
        Minimum number of business days of history required

    Returns:
    --------
    list : Stock tickers that have sufficient history
    """
    min_required_date = current_date - BDay(lookback_days)
    first_valid_dates = data[all_stocks].apply(lambda x: x.first_valid_index())
    eligible = first_valid_dates[first_valid_dates <= min_required_date].index.tolist()
    return eligible

# --- Sector Mapping for 42 Stocks ---
SECTOR_MAPPING = {
    # Technology: 6 stocks
    'NVDA': 'Technology', 'MSFT': 'Technology', 'AAPL': 'Technology',
    'AVGO': 'Technology', 'ORCL': 'Technology', 'PLTR': 'Technology',
    # Financials: 6 stocks
    'BRK.B': 'Financials', 'JPM': 'Financials', 'V': 'Financials',
    'BAC': 'Financials', 'MA': 'Financials', 'GS': 'Financials',
    # Healthcare: 5 stocks
    'LLY': 'Healthcare', 'JNJ': 'Healthcare', 'ABBV': 'Healthcare',
    'UNH': 'Healthcare', 'MRK': 'Healthcare',
    # Consumer Cyclical: 4 stocks
    'AMZN': 'Consumer Cyclical', 'TSLA': 'Consumer Cyclical',
    'HD': 'Consumer Cyclical', 'MCD': 'Consumer Cyclical',
    # Industrials: 4 stocks
    'GE': 'Industrials', 'CAT': 'Industrials', 'RTX': 'Industrials', 'GEV': 'Industrials',
    # Consumer Defensive: 4 stocks
    'WMT': 'Consumer Defensive', 'PG': 'Consumer Defensive',
    'COST': 'Consumer Defensive', 'KO': 'Consumer Defensive',
    # Communication Services: 3 stocks
    'GOOGL': 'Communication', 'META': 'Communication', 'NFLX': 'Communication',
    # Energy: 3 stocks
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
    # Materials: 2 stocks
    'LIN': 'Materials', 'APD': 'Materials',
    # Utilities: 2 stocks
    'NEE': 'Utilities', 'SO': 'Utilities',
    # Real Estate: 2 stocks
    'PLD': 'Real Estate', 'AMT': 'Real Estate',
}

def calculate_sector_exposure(allocation, prices):
    """
    Calculate portfolio value distribution by sector.

    Parameters:
    -----------
    allocation : dict
        {ticker: shares} mapping
    prices : dict
        {ticker: price} mapping

    Returns:
    --------
    dict : {sector: weight_percentage}
    """
    total_value = sum(allocation.get(t, 0) * prices.get(t, 0) for t in allocation)
    if total_value == 0:
        return {}

    sector_values = {}
    for ticker, shares in allocation.items():
        sector = SECTOR_MAPPING.get(ticker, 'Unknown')
        value = shares * prices.get(ticker, 0)
        sector_values[sector] = sector_values.get(sector, 0) + value

    return {sector: (value / total_value) * 100 for sector, value in sector_values.items()}

def calculate_sector_concentration(sector_weights):
    """
    Calculate Herfindahl-Hirschman Index for sector concentration.

    HHI = sum of squared weights (in decimal form) * 100
    - 100 = single sector (maximum concentration)
    - <15 = diversified (per DOJ antitrust guidelines)
    - 15-25 = moderate concentration
    - >25 = high concentration

    Parameters:
    -----------
    sector_weights : dict
        {sector: weight_percentage}

    Returns:
    --------
    float : HHI value
    """
    weights_squared = [(w/100)**2 for w in sector_weights.values() if w > 0]
    hhi = sum(weights_squared) * 100
    return hhi

# 1. --- Configuration ---
data = load_data(path="SP500_42stocks_baseline_adjusted_close.csv")

# Ensure all data columns are numeric, coercing errors
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

sim_start_date = "2020-07-01"  # Q3 2020 start (5-year backtest)
sim_end_date = "2025-08-01"    # 5 years total

rebalance_freq = '63B'  # Quarterly rebalancing
training_lookback_days = 120
risk_lookback_period = 120
exchange = 'NYSE'

initial_budget = 20000
new_invest_per_period = 0  # No DCA for fair comparison
benchmark_assets = ['SPY']

# Full potential universe (all stocks except benchmark)
trading_universe_full = list(data.columns.drop(benchmark_assets))

print(f"All available assets: {data.columns.tolist()}")
print(f"Total assets in data: {len(data.columns)}")
print(f"Potential trading universe: {len(trading_universe_full)} stocks (excluding {benchmark_assets} benchmark)")

# Keep all data (no pre-filtering - dynamic filtering happens at each rebalance)
all_needed_assets = trading_universe_full + benchmark_assets
data = data[all_needed_assets]

print(f"\nUsing DYNAMIC UNIVERSE approach:")
print(f"  - Stocks become eligible when they have {training_lookback_days + risk_lookback_period} days of history")
print(f"  - Universe checked at each rebalance (every {rebalance_freq})")
print(f"  - Follows MSCI index methodology for new stock inclusion\n")

# --- Optimization Parameters ---
k = 2
lambda1 = 1E12
q = 0.2
H_scale = 1E5
solver_type = 'classic'

# --- Risk Manager Parameters ---
sharpe_n = 3
confidence_level = 0.95

# ACADEMIC RESEARCH VERSION:
# - No cash-out logic (portfolio stays fully invested like SPY)
# - Risk managed through optimization (q parameter penalizes volatility)
# - CVaR calculated for monitoring/analysis only, not for trading decisions
# - Pure systematic rules, zero discretionary decisions

# 2. --- Setup with Market Calendar ---
market_cal = mcal.get_calendar(exchange)
data_start_date_for_calendar = (pd.to_datetime(sim_start_date) - BDay(training_lookback_days + risk_lookback_period + 5)).strftime('%Y-%m-%d')
full_schedule = market_cal.schedule(start_date=data_start_date_for_calendar, end_date=sim_end_date)
all_trading_days = full_schedule.index

approx_rebalance_dates = pd.date_range(start=sim_start_date, end=sim_end_date, freq=rebalance_freq)
rebalance_indices = all_trading_days.searchsorted(approx_rebalance_dates, side='left')
rebalance_indices = rebalance_indices[rebalance_indices < len(all_trading_days)]
actual_rebalance_dates = all_trading_days[rebalance_indices]

# Get initial eligible universe
initial_train_end_loc = all_trading_days.get_loc(actual_rebalance_dates[0]) - 1
initial_train_end = all_trading_days[initial_train_end_loc]
initial_train_start_loc = initial_train_end_loc - training_lookback_days
if initial_train_start_loc < 0: initial_train_start_loc = 0
initial_train_start = all_trading_days[initial_train_start_loc]

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
data_for_initial_screening = data[initial_eligible_stocks]
initial_sharpe_results = SharpeRatioCalculator(data_for_initial_screening, initial_train_start.strftime('%Y-%m-%d'),
initial_train_end.strftime('%Y-%m-%d'), risk_free=0.0, print_out=False)

initial_positive_mean_returns = initial_sharpe_results['r_i_series'][initial_sharpe_results['r_i_series'] > 0]
if not initial_positive_mean_returns.empty:
    initial_candidate_sharpes = initial_sharpe_results['sharpe_series'][initial_positive_mean_returns.index]
    initial_assets = initial_candidate_sharpes.dropna().sort_values(ascending=False).head(sharpe_n).index.tolist()
else:
    initial_assets = []

# Initialize PortfolioManager with initial eligible universe
PM = PortfolioManager(data=data[initial_eligible_stocks],
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
is_first_rebalance = True
previous_eligible_stocks = initial_eligible_stocks

# --- Tracking Variables for Enhanced Analytics ---
composition_history = []
turnover_history = []
previous_allocation = {}
sector_exposure_history = []
concentration_history = []

simulation_days = all_trading_days[all_trading_days.slice_indexer(sim_start_date, sim_end_date)]

print(f"\n--- Starting Hybrid Quantum-Classical Simulation Using '{exchange}' Calendar ---")
for i, current_date in enumerate(simulation_days):
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
            if cvar_95 < -0.10:
                print(f"  [RISK MONITOR] {current_date.strftime('%Y-%m-%d')}: CVaR = {cvar_95:.2%}, VaR = {var_95:.2%}")
        except Exception as e:
            pass  # Silent failure for monitoring

    # --- Rebalancing Logic (Only on Scheduled Dates) ---
    is_regular_rebalance = current_date in actual_rebalance_dates

    if is_regular_rebalance:
        all_rebalance_events.append(current_date)
        current_budget_for_opt = current_portfolio_value

        # Only add new investment after the first rebalance
        if not is_first_rebalance:
            current_budget_for_opt += new_invest_per_period
            print(f"--- Rebalancing on {current_date.strftime('%Y-%m-%d')} --- (Budget: ${current_portfolio_value:.2f} + New Investment: ${new_invest_per_period:.2f} = ${current_budget_for_opt:.2f})")
        else:
            print(f"--- First Rebalancing on {current_date.strftime('%Y-%m-%d')} --- (Initial Budget: ${current_budget_for_opt:.2f})")
            is_first_rebalance = False

        current_date_loc = all_trading_days.get_loc(current_date)
        train_end_loc = current_date_loc - 1
        if train_end_loc < 0:
            print("Not enough historical data to train. Skipping rebalance.")
            continue
        train_end = all_trading_days[train_end_loc]
        train_start_loc = train_end_loc - training_lookback_days
        if train_start_loc < 0: train_start_loc = 0
        train_start = all_trading_days[train_start_loc]

        # --- DYNAMIC UNIVERSE: Update eligible stocks based on data availability ---
        current_eligible_stocks = get_eligible_stocks(
            current_date=train_end,
            all_stocks=trading_universe_full,
            data=data,
            lookback_days=training_lookback_days + risk_lookback_period
        )

        # Check for universe expansion
        if len(current_eligible_stocks) > len(previous_eligible_stocks):
            new_stocks = set(current_eligible_stocks) - set(previous_eligible_stocks)
            print(f"  [UNIVERSE EXPANSION] +{len(new_stocks)} new stocks now eligible: {sorted(new_stocks)}, Total eligible stocks: {len(current_eligible_stocks)}")
            previous_eligible_stocks = current_eligible_stocks

        # Screen assets from current eligible universe
        data_for_screening = data[current_eligible_stocks]
        current_sharpe_results = SharpeRatioCalculator(data_for_screening, train_start.strftime('%Y-%m-%d'),
                                                        train_end.strftime('%Y-%m-%d'), risk_free=0.0, print_out=False)

        # Select top N assets from eligible universe
        positive_mean_returns = current_sharpe_results['r_i_series'][current_sharpe_results['r_i_series'] > 0]
        if not positive_mean_returns.empty:
            candidate_sharpes = current_sharpe_results['sharpe_series'][positive_mean_returns.index]
            selected_assets = candidate_sharpes.dropna().sort_values(ascending=False).head(sharpe_n).index.tolist()
        else:
            selected_assets = []

        # Update PM with selected assets
        PM.assets_portfolio = selected_assets

        print(f"Screening for top assets between {train_start.strftime('%Y-%m-%d')} and {train_end.strftime('%Y-%m-%d')}...")
        print(f"New top assets selected: {selected_assets}")

        if not PM.assets_portfolio:
            print("--- No assets with positive Sharpe. Maintaining previous allocation. ---")
            if last_allocation:
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
                current_portfolio_value = sum(
                    last_allocation.get(ticker, 0) * data.loc[current_date, ticker]
                    for ticker in last_allocation if ticker in data.columns
                )
                portfolio_history[-1]['value'] = current_portfolio_value
            continue

        print(f"  [DEBUG] Assets passed to optimizer: {PM.assets_portfolio}")

        # Update PM data with new eligible stocks
        PM.data = data[PM.assets_portfolio]

        # ============================================================
        # HYBRID WORKFLOW: QAOA → Classical Refinement
        # ============================================================

        # --- Stage 1: Quantum Optimization (Coarse Discrete Solution) ---
        print("\n  [STAGE 1] Running Quantum Optimization (QAOA-like)...")
        opt_result_quantum = PM.run_single_optimization(
            current_date=current_date.strftime('%Y-%m-%d'),
            train_start=train_start.strftime('%Y-%m-%d'),
            train_end=train_end.strftime('%Y-%m-%d'),
            budget=current_budget_for_opt,
            k=k, lambda1=lambda1, q=q, H_scale=H_scale, solver_type=solver_type,
            latest_open_prices=data.loc[current_date].to_dict()
        )

        if not opt_result_quantum or not opt_result_quantum.get('allocation'):
            print("  [STAGE 1 FAILED] Quantum optimization returned no allocation. Skipping refinement.")
            last_allocation = {}
            portfolio_history[-1]['allocation'] = {}
            portfolio_history[-1]['value'] = current_portfolio_value
            continue

        print(f"  [STAGE 1 COMPLETE] Quantum Portfolio Value: ${opt_result_quantum['value']:.2f}")
        print(f"  [STAGE 1 ALLOCATION] {opt_result_quantum['allocation']}")

        # --- Stage 2: Classical Refinement (High-Precision Continuous Solution) ---
        print("\n  [STAGE 2] Running Classical Refinement (SLSQP)...")

        # Subset covariance matrix and returns to only selected assets (critical fix!)
        selected_returns = current_sharpe_results['r_i_series'].loc[PM.assets_portfolio]
        selected_cov_indices = [current_sharpe_results['r_i_series'].index.get_loc(asset) for asset in PM.assets_portfolio]
        selected_cov_matrix = current_sharpe_results['sigma'][np.ix_(selected_cov_indices, selected_cov_indices)]

        refined_result = refine_portfolio(
            qaoa_result=opt_result_quantum,
            latest_prices=data.loc[current_date],
            expected_returns=selected_returns,
            cov_matrix=selected_cov_matrix,
            q_risk_factor=q,
            budget=current_budget_for_opt,
            assets=PM.assets_portfolio
        )

        print(f"  [STAGE 2 COMPLETE] Refined Portfolio Value: ${refined_result['value']:.2f}")
        print(f"  [STAGE 2 ALLOCATION] {refined_result['allocation']}")

        # Calculate improvement
        improvement = refined_result['value'] - opt_result_quantum['value']
        improvement_pct = (improvement / opt_result_quantum['value']) * 100 if opt_result_quantum['value'] > 0 else 0
        print(f"  [HYBRID IMPROVEMENT] ${improvement:.2f} ({improvement_pct:+.2f}%)\n")

        # Use refined result for portfolio
        last_allocation = refined_result['allocation']
        portfolio_history[-1]['allocation'] = last_allocation
        portfolio_history[-1]['value'] = refined_result['value']
        portfolio_high_value = refined_result['value']

        # --- Track Portfolio Composition ---
        composition_history.append({
            'date': current_date,
            'assets': selected_assets.copy(),
            'num_assets': len(last_allocation),
            'allocation_dict': last_allocation.copy()
        })

        # --- Calculate and Track Turnover ---
        if previous_allocation:
            old_value = sum(previous_allocation.get(t, 0) * data.loc[current_date, t]
                          for t in previous_allocation if t in data.columns)
            new_value = sum(last_allocation.get(t, 0) * data.loc[current_date, t]
                          for t in last_allocation if t in data.columns)

            turnover = 0
            all_tickers = set(list(previous_allocation.keys()) + list(last_allocation.keys()))
            for ticker in all_tickers:
                if ticker not in data.columns:
                    continue
                old_weight = (previous_allocation.get(ticker, 0) * data.loc[current_date, ticker] / old_value) if old_value > 0 else 0
                new_weight = (last_allocation.get(ticker, 0) * data.loc[current_date, ticker] / new_value) if new_value > 0 else 0
                turnover += abs(new_weight - old_weight)

            turnover_history.append({'date': current_date, 'turnover': turnover * 100})
            print(f"  Portfolio Turnover: {turnover*100:.2f}%")

        previous_allocation = last_allocation.copy()

        # --- Track Sector Exposure ---
        if last_allocation:
            sector_exp = calculate_sector_exposure(last_allocation, data.loc[current_date].to_dict())
            sector_exposure_history.append({'date': current_date, **sector_exp})

            hhi = calculate_sector_concentration(sector_exp)
            concentration_history.append({
                'date': current_date,
                'HHI': hhi,
                'num_sectors': len(sector_exp)
            })

            print(f"  Sector Exposure: {', '.join([f'{s}: {w:.1f}%' for s, w in sorted(sector_exp.items(), key=lambda x: x[1], reverse=True)])}")
            print(f"  Sector HHI: {hhi:.1f} ({len(sector_exp)} sectors)")

        print(f"End of Rebalance Period. Refined Portfolio Value: ${refined_result['value']:.2f}")

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
    invested_series_opt.iloc[0] = initial_budget
    for idx, date in enumerate(actual_rebalance_dates):
        if date in invested_series_opt.index and idx > 0:
            invested_series_opt[date] += new_invest_per_period
    invested_series_opt = invested_series_opt.cumsum()

roi_optimized = ((opt_res_df - invested_series_opt) / invested_series_opt) * 100

# --- Process SPY Benchmark Results (Daily Tracking) ---
from qfolio.analysis.DailyBenchmarkSimulator import DailyBenchmarkSimulator
spy_benchmark = DailyBenchmarkSimulator(
    assets=['SPY'],
    data=data,
    init_budget=initial_budget,
    new_investment=new_invest_per_period
)

spy_daily_values, spy_daily_roi = spy_benchmark.simulate(
    start_date=sim_start_date,
    end_date=sim_end_date,
    rebalance_dates=actual_rebalance_dates
)

spy_bm_results = spy_daily_roi['SPY']
spy_values = spy_daily_values['SPY']

# --- Calculate Performance Statistics ---
opt_daily_returns = opt_res_df.pct_change().dropna()
opt_total_return = roi_optimized.iloc[-1]
opt_volatility = opt_daily_returns.std() * np.sqrt(252) * 100
opt_sharpe = (opt_daily_returns.mean() / opt_daily_returns.std()) * np.sqrt(252) if opt_daily_returns.std() > 0 else 0

opt_cumulative = (1 + opt_daily_returns).cumprod()
opt_running_max = opt_cumulative.cummax()
opt_drawdown = (opt_cumulative - opt_running_max) / opt_running_max
opt_max_drawdown = opt_drawdown.min() * 100

opt_win_rate = (opt_daily_returns > 0).sum() / len(opt_daily_returns) * 100 if len(opt_daily_returns) > 0 else 0

days_in_market = sum(1 for record in portfolio_history if len(record['allocation']) > 0)
total_days = len(portfolio_history)

spy_stats = spy_benchmark.get_statistics(spy_daily_values, spy_daily_roi)['SPY']
spy_daily_returns = spy_daily_values['SPY'].pct_change().dropna()
spy_total_return = spy_stats['Total Return (%)']
spy_volatility = spy_stats['Volatility (%)']
spy_sharpe = spy_stats['Sharpe Ratio']
spy_max_drawdown = spy_stats['Max Drawdown (%)']
spy_win_rate = spy_stats['Win Rate (%)']

# --- Upside/Downside Capture Ratios ---
market_up_days = spy_daily_returns > 0
market_down_days = spy_daily_returns < 0

if market_up_days.sum() > 0:
    opt_returns_market_up = opt_daily_returns[market_up_days].mean() * 252
    spy_returns_market_up = spy_daily_returns[market_up_days].mean() * 252
    upside_capture = (opt_returns_market_up / spy_returns_market_up) * 100 if spy_returns_market_up != 0 else 0
else:
    upside_capture = 0

if market_down_days.sum() > 0:
    opt_returns_market_down = opt_daily_returns[market_down_days].mean() * 252
    spy_returns_market_down = spy_daily_returns[market_down_days].mean() * 252
    downside_capture = (opt_returns_market_down / spy_returns_market_down) * 100 if spy_returns_market_down != 0 else 0
else:
    downside_capture = 0

capture_ratio = upside_capture / abs(downside_capture) if downside_capture != 0 else 0

# --- Drawdown Duration Analysis ---
underwater = opt_drawdown < 0
if underwater.any():
    underwater_groups = (underwater != underwater.shift()).cumsum()
    drawdown_durations = underwater[underwater].groupby(underwater_groups).size()
    max_drawdown_days = drawdown_durations.max()
    avg_drawdown_days = drawdown_durations.mean()
else:
    max_drawdown_days = 0
    avg_drawdown_days = 0

spy_cumulative = (1 + spy_daily_returns).cumprod()
spy_running_max = spy_cumulative.cummax()
spy_drawdown = (spy_cumulative - spy_running_max) / spy_running_max
spy_underwater = spy_drawdown < 0
if spy_underwater.any():
    spy_underwater_groups = (spy_underwater != spy_underwater.shift()).cumsum()
    spy_drawdown_durations = spy_underwater[spy_underwater].groupby(spy_underwater_groups).size()
    spy_max_drawdown_days = spy_drawdown_durations.max()
    spy_avg_drawdown_days = spy_drawdown_durations.mean()
else:
    spy_max_drawdown_days = 0
    spy_avg_drawdown_days = 0

# --- QuantStats Comprehensive Metrics ---
greeks = qs.stats.greeks(opt_daily_returns, spy_daily_returns)
opt_alpha = greeks['alpha'] * 252 * 100
opt_beta = greeks['beta']
opt_r2 = qs.stats.r_squared(opt_daily_returns, spy_daily_returns)
opt_treynor = (opt_daily_returns.mean() * 252) / opt_beta if opt_beta != 0 else 0
opt_sortino = qs.stats.sortino(opt_daily_returns)
opt_calmar = qs.stats.calmar(opt_daily_returns)
opt_cagr = qs.stats.cagr(opt_daily_returns) * 100
opt_information_ratio = qs.stats.information_ratio(opt_daily_returns, spy_daily_returns)
opt_omega = qs.stats.omega(opt_daily_returns)
opt_gain_to_pain = qs.stats.gain_to_pain_ratio(opt_daily_returns)
opt_payoff = qs.stats.payoff_ratio(opt_daily_returns)
opt_profit_factor = qs.stats.profit_factor(opt_daily_returns)
opt_recovery_factor = qs.stats.recovery_factor(opt_daily_returns)
opt_tail_ratio = qs.stats.tail_ratio(opt_daily_returns)
opt_skew = qs.stats.skew(opt_daily_returns)
opt_kurtosis = qs.stats.kurtosis(opt_daily_returns)

spy_greeks = qs.stats.greeks(spy_daily_returns, spy_daily_returns)
spy_alpha = spy_greeks['alpha'] * 252 * 100
spy_beta = spy_greeks['beta']
spy_r2 = qs.stats.r_squared(spy_daily_returns, spy_daily_returns)
spy_treynor = (spy_daily_returns.mean() * 252) / spy_beta if spy_beta != 0 else 0
spy_sortino = qs.stats.sortino(spy_daily_returns)
spy_calmar = qs.stats.calmar(spy_daily_returns)
spy_cagr = qs.stats.cagr(spy_daily_returns) * 100
spy_information_ratio = qs.stats.information_ratio(spy_daily_returns, spy_daily_returns)
spy_omega = qs.stats.omega(spy_daily_returns)
spy_gain_to_pain = qs.stats.gain_to_pain_ratio(spy_daily_returns)
spy_payoff = qs.stats.payoff_ratio(spy_daily_returns)
spy_profit_factor = qs.stats.profit_factor(spy_daily_returns)
spy_recovery_factor = qs.stats.recovery_factor(spy_daily_returns)
spy_tail_ratio = qs.stats.tail_ratio(spy_daily_returns)
spy_skew = qs.stats.skew(spy_daily_returns)
spy_kurtosis = qs.stats.kurtosis(spy_daily_returns)

# --- Statistical Significance Tests ---
excess_returns = opt_daily_returns - spy_daily_returns
t_stat, p_value = stats.ttest_1samp(excess_returns.dropna(), 0)

if p_value < 0.01:
    significance = "*** (Highly Significant)"
elif p_value < 0.05:
    significance = "** (Significant)"
elif p_value < 0.1:
    significance = "* (Marginally Significant)"
else:
    significance = "(Not Significant)"

# --- Print Comprehensive Comparison ---
print("\n" + "="*90)
print(" " * 20 + "HYBRID QUANTUM-CLASSICAL PERFORMANCE COMPARISON")
print("="*90)
print(f"\n{'Metric':<40} {'Hybrid Portfolio':<22} {'SPY Benchmark':<22}")
print("-"*90)

print("\n" + "RETURNS & GROWTH".center(90, " "))
print(f"{'Total Return (%)':<40} {opt_total_return:>20.2f}% {spy_total_return:>20.2f}%")
print(f"{'CAGR (%)':<40} {opt_cagr:>20.2f}% {spy_cagr:>20.2f}%")
print(f"{'Alpha (annualized %)':<40} {opt_alpha:>20.2f}% {spy_alpha:>20.2f}%")

print("\n" + "RISK METRICS".center(90, " "))
print(f"{'Volatility (annual %)':<40} {opt_volatility:>20.2f}% {spy_volatility:>20.2f}%")
print(f"{'Beta':<40} {opt_beta:>20.2f}  {spy_beta:>20.2f}")
print(f"{'R-Squared':<40} {opt_r2:>20.2f}  {spy_r2:>20.2f}")
print(f"{'Max Drawdown (%)':<40} {opt_max_drawdown:>20.2f}% {spy_max_drawdown:>20.2f}%")

print("\n" + "RISK-ADJUSTED RETURNS".center(90, " "))
print(f"{'Sharpe Ratio':<40} {opt_sharpe:>20.2f}  {spy_sharpe:>20.2f}")
print(f"{'Sortino Ratio':<40} {opt_sortino:>20.2f}  {spy_sortino:>20.2f}")
print(f"{'Calmar Ratio':<40} {opt_calmar:>20.2f}  {spy_calmar:>20.2f}")
print(f"{'Treynor Ratio':<40} {opt_treynor:>20.2f}  {spy_treynor:>20.2f}")
print(f"{'Information Ratio':<40} {opt_information_ratio:>20.2f}  {'N/A':>20}")

print("\n" + "PERFORMANCE RATIOS".center(90, " "))
print(f"{'Omega Ratio':<40} {opt_omega:>20.2f}  {spy_omega:>20.2f}")
print(f"{'Profit Factor':<40} {opt_profit_factor:>20.2f}  {spy_profit_factor:>20.2f}")
print(f"{'Gain-to-Pain Ratio':<40} {opt_gain_to_pain:>20.2f}  {spy_gain_to_pain:>20.2f}")
print(f"{'Recovery Factor':<40} {opt_recovery_factor:>20.2f}  {spy_recovery_factor:>20.2f}")
print(f"{'Payoff Ratio':<40} {opt_payoff:>20.2f}  {spy_payoff:>20.2f}")

print("\n" + "WIN/LOSS STATISTICS".center(90, " "))
print(f"{'Win Rate (%)':<40} {opt_win_rate:>20.2f}% {spy_win_rate:>20.2f}%")
print(f"{'Tail Ratio':<40} {opt_tail_ratio:>20.2f}  {spy_tail_ratio:>20.2f}")

print("\n" + "RETURN DISTRIBUTION".center(90, " "))
print(f"{'Skewness':<40} {opt_skew:>20.2f}  {spy_skew:>20.2f}")
print(f"{'Kurtosis':<40} {opt_kurtosis:>20.2f}  {spy_kurtosis:>20.2f}")

print("\n" + "MARKET CAPTURE".center(90, " "))
print(f"{'Upside Capture (%)':<40} {upside_capture:>20.2f}% {'100.00%':>20}")
print(f"{'Downside Capture (%)':<40} {abs(downside_capture):>20.2f}% {'100.00%':>20}")
print(f"{'Capture Ratio':<40} {capture_ratio:>20.2f}  {'1.00':>20}")

print("\n" + "DRAWDOWN ANALYSIS".center(90, " "))
print(f"{'Max Drawdown Duration (days)':<40} {max_drawdown_days:>20.0f}  {spy_max_drawdown_days:>20.0f}")
if avg_drawdown_days > 0:
    print(f"{'Avg Drawdown Duration (days)':<40} {avg_drawdown_days:>20.1f}  {spy_avg_drawdown_days:>20.1f}")

print("\n" + "PORTFOLIO CHARACTERISTICS".center(90, " "))
if turnover_history:
    avg_turnover = sum(t['turnover'] for t in turnover_history) / len(turnover_history)
    print(f"{'Average Turnover per Rebalance (%)':<40} {avg_turnover:>20.2f}% {'0.00%':>20}")
print(f"{'Number of Rebalances':<40} {len(actual_rebalance_dates):>20}  {len(actual_rebalance_dates):>20}")
if composition_history:
    avg_assets = sum(c['num_assets'] for c in composition_history) / len(composition_history)
    print(f"{'Avg Assets per Period':<40} {avg_assets:>20.1f}  {'1.0':>20}")

if concentration_history:
    avg_hhi = sum(c['HHI'] for c in concentration_history) / len(concentration_history)
    avg_sectors = sum(c['num_sectors'] for c in concentration_history) / len(concentration_history)
    print("\n" + "SECTOR DIVERSIFICATION".center(90, " "))
    print(f"{'Average Sector HHI':<40} {avg_hhi:>20.1f}  {'N/A':>20}")
    hhi_interpretation = "(Highly Concentrated)" if avg_hhi > 25 else "(Moderate)" if avg_hhi > 15 else "(Diversified)"
    print(f"{'HHI Interpretation':<40} {hhi_interpretation:>42}")
    print(f"{'Avg Sectors Held per Period':<40} {avg_sectors:>20.1f}  {'11 (all)':>20}")

print("\n" + "STATISTICAL SIGNIFICANCE".center(90, " "))
print(f"{'Excess Return t-statistic':<40} {t_stat:>20.4f}  {'N/A':>20}")
print(f"{'Excess Return p-value':<40} {p_value:>20.4f}  {'N/A':>20}")
print(f"{'Significance Level':<40} {significance:>42}")

print("-"*90)

roi_diff = opt_total_return - spy_total_return
sharpe_diff = opt_sharpe - spy_sharpe
print(f"\n{'Performance vs SPY:':<30}")
print(f"{'  Return Difference':<30} {roi_diff:>18.2f}%  {'(Better)' if roi_diff > 0 else '(Worse)'}")
print(f"{'  Sharpe Difference':<30} {sharpe_diff:>18.2f}   {'(Better)' if sharpe_diff > 0 else '(Worse)'}")
print("="*90 + "\n")

# --- Plotting ROI ---
plt.figure(figsize=(12, 6))
plt.plot(roi_optimized, label="Hybrid Quantum-Classical ROI", linewidth=2, color='purple')
plt.plot(spy_bm_results, label="100% SPY Benchmark ROI", linewidth=2, linestyle='--', color='blue')

if all_rebalance_events:
    for i, date in enumerate(all_rebalance_events):
        plt.axvline(x=date, color='red', linestyle='--', linewidth=1, label='Rebalance Date' if i == 0 else "")

plt.title(f"Hybrid Quantum-Classical Portfolio ROI vs. SPY Benchmark ({exchange})")
plt.xlabel("Date")
plt.ylabel("Return on Investment (%)")
plt.axhline(0, color='grey', linestyle='--')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Daily Value Comparison Chart ---
plt.figure(figsize=(14, 7))
plt.plot(opt_res_df, label="Hybrid Quantum-Classical Portfolio", linewidth=2.5, color='purple')
plt.plot(spy_values, label="SPY Buy-and-Hold", linewidth=2.5, linestyle='--', color='blue')

plt.axhline(y=initial_budget, color='gray', linestyle=':', alpha=0.5, linewidth=1, label=f'Initial: ${initial_budget}')

for i, date in enumerate(actual_rebalance_dates):
    if date in opt_res_df.index:
        plt.axvline(x=date, color='red', alpha=0.2, linestyle='--', linewidth=0.8,
                   label='Rebalance Date' if i == 0 else "")

opt_beats_spy = opt_res_df > spy_values
if opt_beats_spy.any():
    plt.fill_between(opt_res_df.index, opt_res_df, spy_values,
                     where=opt_beats_spy, alpha=0.2, color='green',
                     label='Hybrid Outperforms', interpolate=True)
    plt.fill_between(opt_res_df.index, opt_res_df, spy_values,
                     where=~opt_beats_spy, alpha=0.2, color='red',
                     label='SPY Outperforms', interpolate=True)

plt.title(f"Hybrid Quantum-Classical vs SPY ({exchange})\n" +
          f"Stage 1: Quantum QAOA | Stage 2: Classical SLSQP Refinement",
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

# --- Portfolio Composition & Turnover Over Time ---
if composition_history and turnover_history:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    comp_df = pd.DataFrame(composition_history)
    comp_df['date'] = pd.to_datetime(comp_df['date'])
    comp_df = comp_df.set_index('date')

    ax1.plot(comp_df.index, comp_df['num_assets'], linewidth=2, color='darkblue', marker='o', markersize=6)
    ax1.set_title("Portfolio Size Over Time (Hybrid Strategy)", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Number of Assets", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=sharpe_n, color='red', linestyle='--', alpha=0.5, linewidth=2, label=f'Target: {sharpe_n}')
    ax1.legend()
    ax1.set_ylim(bottom=0)

    turnover_df = pd.DataFrame(turnover_history)
    turnover_df['date'] = pd.to_datetime(turnover_df['date'])
    turnover_df = turnover_df.set_index('date')

    ax2.bar(turnover_df.index, turnover_df['turnover'], width=20, color='orange', alpha=0.7, edgecolor='black')
    ax2.set_title("Portfolio Turnover at Each Rebalance (Hybrid Strategy)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Date", fontsize=12)
    ax2.set_ylabel("Turnover (%)", fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=50, color='red', linestyle='--', alpha=0.3, linewidth=2, label='50% threshold')
    ax2.legend()

    plt.tight_layout()
    plt.show()

# --- Sector Exposure Stacked Area Chart ---
if sector_exposure_history:
    sector_df = pd.DataFrame(sector_exposure_history)
    sector_df['date'] = pd.to_datetime(sector_df['date'])
    sector_df = sector_df.set_index('date')
    sector_df = sector_df.fillna(0)

    sector_totals = sector_df.sum().sort_values(ascending=False)
    sector_df_sorted = sector_df[sector_totals.index]

    plt.figure(figsize=(14, 8))
    sector_df_sorted.plot.area(stacked=True, alpha=0.7, figsize=(14, 8),
                                colormap='tab20', linewidth=0, ax=plt.gca())
    plt.title("Sector Allocation Over Time (Hybrid Quantum-Classical)",
              fontsize=14, fontweight='bold')
    plt.ylabel("Portfolio Weight (%)", fontsize=12)
    plt.xlabel("Rebalance Date", fontsize=12)
    plt.legend(title="Sector", bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=9)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

# --- Sector Exposure Heatmap ---
if sector_exposure_history:
    sector_df = pd.DataFrame(sector_exposure_history)
    sector_df['date'] = pd.to_datetime(sector_df['date'])
    sector_df = sector_df.set_index('date')
    sector_df = sector_df.fillna(0)

    sector_totals = sector_df.sum().sort_values(ascending=False)
    sector_df_sorted = sector_df[sector_totals.index]

    plt.figure(figsize=(16, 8))
    sns.heatmap(sector_df_sorted.T, cmap='YlOrRd', cbar_kws={'label': 'Allocation (%)'},
                linewidths=0.5, linecolor='lightgray', vmin=0, vmax=100)
    plt.title("Sector Exposure Heatmap (Hybrid Strategy)", fontsize=14, fontweight='bold')
    plt.xlabel("Rebalance Date", fontsize=12)
    plt.ylabel("Sector", fontsize=12)

    n_dates = len(sector_df_sorted)
    step = max(1, n_dates // 10)
    plt.xticks(range(0, n_dates, step),
               [sector_df_sorted.index[i].strftime('%Y-%m-%d') for i in range(0, n_dates, step)],
               rotation=45)

    plt.tight_layout()
    plt.show()

# --- Sector Concentration (HHI) Over Time ---
if concentration_history:
    conc_df = pd.DataFrame(concentration_history)
    conc_df['date'] = pd.to_datetime(conc_df['date'])
    conc_df = conc_df.set_index('date')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    ax1.plot(conc_df.index, conc_df['HHI'], linewidth=2, color='darkred', marker='o', markersize=5)
    ax1.set_title("Sector Concentration (HHI) Over Time (Hybrid Strategy)", fontsize=14, fontweight='bold')
    ax1.set_ylabel("HHI", fontsize=12)
    ax1.axhline(y=25, color='red', linestyle='--', alpha=0.5, linewidth=2, label='High Concentration (>25)')
    ax1.axhline(y=15, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='Moderate (15-25)')
    ax1.fill_between(conc_df.index, 0, 15, alpha=0.1, color='green', label='Diversified (<15)')
    ax1.fill_between(conc_df.index, 15, 25, alpha=0.1, color='orange')
    ax1.fill_between(conc_df.index, 25, 100, alpha=0.1, color='red')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)

    ax2.plot(conc_df.index, conc_df['num_sectors'], linewidth=2, color='darkgreen', marker='s', markersize=5)
    ax2.set_title("Number of Sectors Held Over Time", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Rebalance Date", fontsize=12)
    ax2.set_ylabel("Number of Sectors", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 12)
    ax2.axhline(y=11, color='gray', linestyle=':', alpha=0.5, linewidth=2, label='Max possible (11 sectors)')
    ax2.legend()

    plt.tight_layout()
    plt.show()

# ===== QUANTSTATS HTML REPORTS & DATA EXPORTS =====
print("\n" + "="*90)
print("Generating QuantStats HTML Reports and Exporting Data...")
print("="*90)

opt_returns_for_qs = opt_daily_returns.copy()
spy_returns_for_qs = spy_daily_returns.copy()

common_index = opt_returns_for_qs.index.intersection(spy_returns_for_qs.index)
opt_returns_for_qs = opt_returns_for_qs.loc[common_index]
spy_returns_for_qs = spy_returns_for_qs.loc[common_index]

output_dir = "quantstats_reports_hybrid"
os.makedirs(output_dir, exist_ok=True)

report_filename = f"{output_dir}/portfolio_report_hybrid_{sim_start_date}_{sim_end_date}.html"
print(f"\nGenerating full tearsheet: {report_filename}")

try:
    qs.reports.html(
        opt_returns_for_qs,
        benchmark=spy_returns_for_qs,
        output=report_filename,
        title=f"Hybrid Quantum-Classical Portfolio - Full Tearsheet ({sim_start_date} to {sim_end_date})",
        download_filename=report_filename
    )
    print(f"[OK] Full HTML report saved: {report_filename}")
except Exception as e:
    print(f"[ERROR] Error generating HTML report: {e}")

if composition_history:
    comp_records = []
    for record in composition_history:
        comp_records.append({
            'date': record['date'],
            'num_assets': record['num_assets'],
            'assets': ', '.join(record['assets']),
            'allocation': str(record['allocation_dict'])
        })
    comp_df = pd.DataFrame(comp_records)
    comp_filename = f"{output_dir}/composition_history.csv"
    comp_df.to_csv(comp_filename, index=False)
    print(f"[OK] Composition history saved: {comp_filename}")

if turnover_history:
    turnover_df = pd.DataFrame(turnover_history)
    turnover_filename = f"{output_dir}/turnover_history.csv"
    turnover_df.to_csv(turnover_filename, index=False)
    print(f"[OK] Turnover history saved: {turnover_filename}")

if sector_exposure_history:
    sector_df = pd.DataFrame(sector_exposure_history)
    sector_filename = f"{output_dir}/sector_exposure_history.csv"
    sector_df.to_csv(sector_filename, index=False)
    print(f"[OK] Sector exposure history saved: {sector_filename}")

if concentration_history:
    conc_df = pd.DataFrame(concentration_history)
    conc_filename = f"{output_dir}/concentration_history.csv"
    conc_df.to_csv(conc_filename, index=False)
    print(f"[OK] Concentration history saved: {conc_filename}")

if sector_exposure_history:
    sector_frequency = {}
    sector_total_weight = {}

    for record in sector_exposure_history:
        for sector, weight in record.items():
            if sector == 'date':
                continue
            if weight > 0:
                sector_frequency[sector] = sector_frequency.get(sector, 0) + 1
                sector_total_weight[sector] = sector_total_weight.get(sector, 0) + weight

    total_rebalances = len(sector_exposure_history)
    sector_avg_weight = {s: sector_total_weight[s] / sector_frequency[s] for s in sector_frequency}

    sector_summary = pd.DataFrame({
        'Sector': list(sector_frequency.keys()),
        'Times_Selected': list(sector_frequency.values()),
        'Selection_Rate_Pct': [v/total_rebalances*100 for v in sector_frequency.values()],
        'Avg_Weight_When_Held_Pct': [sector_avg_weight[s] for s in sector_frequency.keys()]
    }).sort_values('Times_Selected', ascending=False)

    sector_summary_filename = f"{output_dir}/sector_frequency_summary.csv"
    sector_summary.to_csv(sector_summary_filename, index=False)
    print(f"[OK] Sector frequency summary saved: {sector_summary_filename}")

    print("\n" + "="*90)
    print(" " * 30 + "SECTOR SELECTION FREQUENCY")
    print("="*90)
    print(sector_summary.to_string(index=False))
    print("="*90)

print("\n" + "="*90)
print("All reports and data exports completed successfully!")
print(f"Output directory: {os.path.abspath(output_dir)}")
print("="*90)

# ===============================================================================
# SAVE RESULTS TO JSON (for grid search analysis)
# ===============================================================================

# Calculate average metrics needed for save_results_module
if composition_history:
    avg_assets = sum(c['num_assets'] for c in composition_history) / len(composition_history)

    # Calculate max and avg single-asset weights
    max_weights = []
    for record in composition_history:
        if record['allocation_dict']:
            current_date_for_weights = record['date']
            prices = data.loc[current_date_for_weights].to_dict()
            total_value = sum(record['allocation_dict'].get(t, 0) * prices.get(t, 0)
                            for t in record['allocation_dict'])
            if total_value > 0:
                weights = [record['allocation_dict'].get(t, 0) * prices.get(t, 0) / total_value * 100
                          for t in record['allocation_dict']]
                max_weights.append(max(weights))

    max_single_asset_weight = max(max_weights) if max_weights else 0.0
    avg_max_asset_weight = sum(max_weights) / len(max_weights) if max_weights else 0.0
else:
    avg_assets = 0.0
    max_single_asset_weight = 0.0
    avg_max_asset_weight = 0.0

if concentration_history:
    avg_hhi = sum(c['HHI'] for c in concentration_history) / len(concentration_history)
    avg_sectors = sum(c['num_sectors'] for c in concentration_history) / len(concentration_history)
else:
    avg_hhi = 0.0
    avg_sectors = 0.0

# Calculate excess returns and outperformance
excess_return = opt_total_return - spy_total_return
outperformance = (opt_total_return / spy_total_return - 1) * 100 if spy_total_return != 0 else 0

# Calculate correlation and tracking error
correlation = opt_daily_returns.corr(spy_daily_returns)
tracking_error = (opt_daily_returns - spy_daily_returns).std() * np.sqrt(252) * 100

# Save results
save_backtest_results(
    # Metadata
    method='hybrid',
    q=q,
    budget=initial_budget,
    lambda1=lambda1,
    k=k,
    solver_type=solver_type,
    sharpe_n=sharpe_n,
    H_scale=H_scale,

    # Simulation config
    sim_start_date=sim_start_date,
    sim_end_date=sim_end_date,
    rebalance_freq=rebalance_freq,
    training_lookback_days=training_lookback_days,
    risk_lookback_period=risk_lookback_period,
    new_invest_per_period=new_invest_per_period,
    exchange=exchange,
    confidence_level=confidence_level,

    # Performance metrics
    total_return_pct=opt_total_return,
    cagr_pct=opt_cagr,
    sharpe_ratio=opt_sharpe,
    sortino_ratio=opt_sortino,
    calmar_ratio=opt_calmar,
    treynor_ratio=opt_treynor,
    information_ratio=opt_information_ratio,
    max_drawdown_pct=opt_max_drawdown,
    volatility_pct=opt_volatility,
    alpha_pct=opt_alpha,
    beta=opt_beta,
    r_squared=opt_r2,
    omega_ratio=opt_omega,
    profit_factor=opt_profit_factor,
    gain_to_pain=opt_gain_to_pain,
    recovery_factor=opt_recovery_factor,
    payoff_ratio=opt_payoff,
    tail_ratio=opt_tail_ratio,
    win_rate_pct=opt_win_rate,
    skewness=opt_skew,
    kurtosis=opt_kurtosis,

    # Market capture
    upside_capture_pct=upside_capture,
    downside_capture_pct=abs(downside_capture),
    capture_ratio=capture_ratio,

    # Portfolio characteristics
    avg_num_assets=avg_assets,
    avg_turnover_pct=avg_turnover,
    num_rebalances=len(actual_rebalance_dates),
    avg_sector_hhi=avg_hhi,
    avg_sectors_held=avg_sectors,
    max_single_asset_weight_pct=max_single_asset_weight,
    avg_max_asset_weight_pct=avg_max_asset_weight,

    # Benchmark comparison
    spy_total_return_pct=spy_total_return,
    excess_return_pct=excess_return,
    outperformance_pct=outperformance,
    correlation=correlation,
    tracking_error_pct=tracking_error,

    # Statistical tests
    t_statistic=t_stat,
    p_value=p_value,
    significance_level=significance,

    # Drawdown analysis
    max_drawdown_duration_days=int(max_drawdown_days),
    avg_drawdown_duration_days=float(avg_drawdown_days),

    # Time series (optional - set False for smaller files)
    dates=[d.strftime('%Y-%m-%d') for d in opt_res_df.index],
    portfolio_values=opt_res_df.values.tolist(),
    daily_returns=opt_daily_returns.values.tolist(),
    drawdown_pct=(opt_drawdown * 100).values.tolist(),

    # Output config
    output_dir='results',
    include_timeseries=False  # Set True if you want full time series
)

print("\n" + "="*90)
print("Results saved for grid search analysis")
print("="*90 + "\n")
