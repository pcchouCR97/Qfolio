"""
Simple Data Loader for Portfolio Optimization
Handles market date adjustments and fixed-window data retrieval.
"""

import pandas as pd
import numpy as np
import pandas_market_calendars as mcal
from pandas.tseries.offsets import BDay
from qfolio.screeners.sharpe_screener import SharpeRatioCalculator

def adjust_to_market_date(date, data, direction='forward'):
    """
    If date doesn't exist in market data, find nearest trading day.

    Args:
        date (str or pd.Timestamp): Target date (e.g., '2024-01-01')
        data (pd.DataFrame): DataFrame with DatetimeIndex
        direction (str): 'forward' (default) or 'backward'

    Returns:
        pd.Timestamp: Adjusted date that exists in market data

    Example:
        >>> adjusted = adjust_to_market_date('2024-01-01', data, direction='forward')
    """
    date = pd.to_datetime(date)

    # If date exists, return it
    if date in data.index:
        return date

    # Find nearest market date
    if direction == 'forward':
        # Find first date >= target
        future_dates = data.index[data.index >= date]
        if len(future_dates) > 0:
            return future_dates[0]
        else:
            raise ValueError(f"No market dates found after {date}")
    else:  # backward
        # Find last date <= target
        past_dates = data.index[data.index <= date]
        if len(past_dates) > 0:
            return past_dates[-1]
        else:
            raise ValueError(f"No market dates found before {date}")


def get_rebalance_dates(start_date, end_date, data, freq='1M'):
    """
    Generate rebalance dates at regular intervals.
    All dates are guaranteed to be market open (auto-adjusted forward if needed).

    Args:
        start_date (str or pd.Timestamp): Start date
        end_date (str or pd.Timestamp): End date
        data (pd.DataFrame): DataFrame with DatetimeIndex
        freq (str): Frequency - supports:
            - '1M', '2M', '3M', etc. (calendar months)
            - '21B', '63B', '252B', etc. (business days)

    Returns:
        list of pd.Timestamp: Rebalance dates (all guaranteed market open)

    Example:
        >>> dates = get_rebalance_dates('2024-01-01', '2024-12-31', data, freq='1M')
    """
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Adjust start/end to market dates
    start_date = adjust_to_market_date(start_date, data, direction='forward')
    end_date = adjust_to_market_date(end_date, data, direction='backward')

    # Generate candidate dates
    if freq.endswith('M'):
        # Calendar month frequency - use 'ME' for month-end
        freq_adjusted = freq.replace('M', 'ME')
        candidate_dates = pd.date_range(start=start_date, end=end_date, freq=freq_adjusted)
    elif freq.endswith('B'):
        # Business day frequency
        num_days = int(freq.replace('B', ''))
        all_trading_days = data.index
        start_idx = all_trading_days.get_loc(start_date)
        end_idx = all_trading_days.get_loc(end_date)

        indices = list(range(start_idx, end_idx + 1, num_days))
        candidate_dates = all_trading_days[indices]
    else:
        raise ValueError(f"Unsupported frequency: {freq}. Use '1M', '2M', or '21B', '63B', etc.")

    # Adjust all candidates to market dates (forward)
    rebalance_dates = []
    for date in candidate_dates:
        if date <= end_date:  # Don't go past end_date
            adjusted = adjust_to_market_date(date, data, direction='forward')
            if adjusted <= end_date and adjusted not in rebalance_dates:
                rebalance_dates.append(adjusted)

    return rebalance_dates


def get_train_data(data, train_end_date, lookback='252B'):
    """
    Get FIXED window of last X business days before train_end_date.
    NO expanding, NO rolling, just a fixed lookback window.

    Args:
        data (pd.DataFrame): Full price data with DatetimeIndex
        train_end_date (str or pd.Timestamp): End date for training window
        lookback (str): Lookback period:
            - '252B' = 1 year of trading days (default)
            - '504B' = 2 years of trading days
            - '63B' = 1 quarter of trading days
            - '21B' = 1 month of trading days

    Returns:
        pd.DataFrame: Price data for the fixed lookback window

    Example:
        >>> # Get last 1 year (252 trading days) before 2024-10-28
        >>> train_prices = get_train_data(data, '2024-10-28', lookback='252B')
    """
    train_end_date = pd.to_datetime(train_end_date)

    # Adjust end date to market date if needed
    train_end_date = adjust_to_market_date(train_end_date, data, direction='backward')

    # Parse lookback
    if not lookback.endswith('B'):
        raise ValueError(f"Lookback must be in business days (e.g., '252B'). Got: {lookback}")

    num_days = int(lookback.replace('B', ''))

    # Get all trading days up to end date
    all_trading_days = data.index[data.index <= train_end_date]

    if len(all_trading_days) < num_days:
        raise ValueError(f"Not enough data. Need {num_days} days, but only {len(all_trading_days)} available before {train_end_date}")

    # Get last num_days trading days
    train_start_idx = len(all_trading_days) - num_days
    train_end_idx = len(all_trading_days)

    train_dates = all_trading_days[train_start_idx:train_end_idx]

    return data.loc[train_dates]

def get_all_trading_date(exchange, sim_start_date, sim_end_date, training_lookback_days, risk_lookback_period):
    """
    Get all trading dates for a given exchange between start_date and end_date.

    Args:
        exchange (str): Exchange code (e.g., 'NYSE', 'NASDAQ')
    """
    
    market_cal = mcal.get_calendar(exchange)

    data_start_date_for_calendar = (pd.to_datetime(sim_start_date) - BDay(training_lookback_days + risk_lookback_period + 5)).strftime('%Y-%m-%d')
    full_schedule = market_cal.schedule(start_date=data_start_date_for_calendar, end_date=sim_end_date)
    all_trading_days = full_schedule.index

    return all_trading_days

def initial_eligible_universe(all_trading_days, actual_rebalance_dates, training_lookback_days):
    """
    Get the initial training window dates based on the first rebalance date.
    Args:
        all_trading_days (pd.DatetimeIndex): All trading days in the dataset
        actual_rebalance_dates (list of pd.Timestamp): List of rebalance dates
        training_lookback_days (int): Number of business days for training lookback
    Returns:
        tuple: (initial_train_start, initial_train_end) as pd.Timestamp
    """
    initial_train_end_loc = all_trading_days.get_loc(actual_rebalance_dates[0]) - 1
    initial_train_end = all_trading_days[initial_train_end_loc]
    initial_train_start_loc = initial_train_end_loc - training_lookback_days
    if initial_train_start_loc < 0: initial_train_start_loc = 0
    initial_train_start = all_trading_days[initial_train_start_loc]

    return initial_train_start, initial_train_end

def get_initial_assets(data, initial_eligible_stocks, initial_train_start, initial_train_end, sharpe_n):
    
    data_for_initial_screening = data[initial_eligible_stocks]
    initial_sharpe_results = SharpeRatioCalculator(data_for_initial_screening, initial_train_start.strftime('%Y-%m-%d'),
    initial_train_end.strftime('%Y-%m-%d'), risk_free=0.0, top_n = sharpe_n, print_out=False)

    initial_positive_mean_returns = initial_sharpe_results['r_i_series'][initial_sharpe_results['r_i_series'] > 0]
    if not initial_positive_mean_returns.empty:
        initial_candidate_sharpes = initial_sharpe_results['sharpe_series'][initial_positive_mean_returns.index]
        initial_assets = initial_candidate_sharpes.dropna().sort_values(ascending=False).head(sharpe_n).index.tolist()
    else:
        initial_assets = []

    return initial_assets