import sys
import numpy as np
import quantstats as qs
import pandas as pd
from qfolio.analysis.DailyBenchmarkSimulator import DailyBenchmarkSimulator

def comparison(portfolio_history,
               data,
               sim_start_date,
               sim_end_date,
               initial_budget: float,
               new_invest_per_period: float,
               actual_rebalance_dates: list):

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

        # Subsequent investments on scheduled rebalance dates (skip first one)
        for idx, date in enumerate(actual_rebalance_dates):
            if date in invested_series_opt.index and idx > 0:  # Skip first rebalance (idx=0)
                invested_series_opt[date] += new_invest_per_period

        invested_series_opt = invested_series_opt.cumsum()

    port_roi = ((opt_res_df - invested_series_opt) / invested_series_opt) * 100
    
    # --- Process SPY Benchmark Results (Daily Tracking) ---
    spy_benchmark = DailyBenchmarkSimulator(
        assets=['SPY'],
        data=data,
        init_budget=initial_budget,
        new_investment=new_invest_per_period
    )

    # Run daily simulation with same rebalance dates as optimized portfolio
    spy_daily_values, spy_daily_roi = spy_benchmark.simulate(
        start_date=sim_start_date,
        end_date=sim_end_date,
        rebalance_dates=actual_rebalance_dates  # Same dates as optimized portfolio!
    )

    # Keep as DataFrame for compatibility with get_statistics()
    # but also return Series for easier plotting
    bm_roi = spy_daily_roi  # Keep DataFrame
    bm_values = spy_daily_values  # Keep DataFrame

    # opt_res_df has all the date data for optimzed one

    port_values = opt_res_df
    return port_roi, port_values, bm_roi, bm_values, spy_benchmark

def up_downside_captrue_ratio(opt_daily_returns, spy_daily_returns):

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

    return upside_capture, downside_capture, capture_ratio

def drawdown_duration(opt_drawdown, spy_daily_returns):
    # For optimized portfolio

    underwater = opt_drawdown < 0
    if underwater.any():
        underwater_groups = (underwater != underwater.shift()).cumsum()
        drawdown_durations = underwater[underwater].groupby(underwater_groups).size()
        max_drawdown_days = drawdown_durations.max()
        avg_drawdown_days = drawdown_durations.mean()
    else:
        max_drawdown_days = 0
        avg_drawdown_days = 0

    # For SPY
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


    return max_drawdown_days, avg_drawdown_days, spy_max_drawdown_days, spy_avg_drawdown_days