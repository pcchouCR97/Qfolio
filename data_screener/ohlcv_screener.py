import pandas as pd
import numpy as np

def SharpeRatioScreener_OHLCV(full_ohlcv_data, train_start, train_end, risk_free=0.0, top_n=3, print_out=True):
    """
    Calculates the Sharpe Ratio for a list of tickers using proper OHLCV data.
    It correctly uses the 'Adj Close' for all return calculations.

    Args:
        full_ohlcv_data (pd.DataFrame): DataFrame with multi-level columns (e.g., ('Open', 'AAPL')).
        train_start (str): The start date for the training period.
        train_end (str): The end date for the training period.
        risk_free (float): The risk-free rate.
        top_n (int): The number of top assets to return.
        print_out (bool): Whether to print the results.

    Returns:
        A dictionary containing the series of Sharpe ratios and mean returns.
    """
    # --- 1. Select only the 'Adj Close' data for return calculations ---
    try:
        adj_close_data = full_ohlcv_data.xs('Adj Close', level=0, axis=1)
    except KeyError:
        # Fallback for older data format if 'Adj Close' is not a level
        adj_close_data = full_ohlcv_data

    # --- 2. Slice the data for the training period ---
    train_data = adj_close_data.loc[train_start:train_end]
    
    # --- 3. Calculate daily returns ---
    daily_returns = train_data.pct_change().dropna()
    
    # --- 4. Calculate Sharpe Ratio ---
    # Get the number of trading days in a year
    trading_days = 252
    
    # Calculate annualized mean returns
    r_i_series = daily_returns.mean() * trading_days
    
    # Calculate annualized volatility (standard deviation of returns)
    volatility_series = daily_returns.std() * np.sqrt(trading_days)
    
    # Calculate Sharpe Ratio
    sharpe_series = (r_i_series - risk_free) / volatility_series
    
    if print_out:
        print("\n--- Sharpe Ratio Screener (OHLCV) Results ---")
        print(sharpe_series.sort_values(ascending=False).to_string())
        print("---------------------------------------------")

    return {
        "sharpe_series": sharpe_series,
        "r_i_series": r_i_series
    }
