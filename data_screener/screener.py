import numpy as np
import pandas as pd
import glob
import sys
import os

# make sure parent directory is on path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qfolio.utils.utilities import *
# or, for specific functions:
# from utilities import SharpeRatioCalculator

#TODO create a class called 

def SharpeRatioCalculator(data, train_start, train_end, risk_free=0.0, top_n=3, print_out=False):
    """
    Compute per-asset Sharpe ratios over [train_start, train_end].
    Keeps original outputs AND adds:
      - per-asset dicts for r_i, sigma_p, Sharpe
      - final line of top-N tickers by Sharpe (default 3)
    No rounding is applied.

    risk_free should match the return interval (e.g., daily if returns are daily).
    """
    # Use your existing function to get returns and r_i, sigma (cov matrix)
    returns, r_i, sigma = StatisticCalculatorRollingD(
        data, train_start, train_end, print_out=False, plot=False
    )

    # Per-asset volatility (std dev of returns); avoid div by zero
    sigma_p_series = returns.std()
    sigma_p = sigma_p_series.values
    sigma_p_nonzero = sigma_p.copy()
    sigma_p_nonzero[sigma_p_nonzero == 0.0] = np.nan  # to avoid inf

    # Mean returns already returned as np.array; align with columns
    r_i_series = pd.Series(r_i, index=returns.columns)

    # Sharpe per asset
    sharpe_series = (r_i_series - risk_free) / sigma_p_series.replace(0.0, np.nan)
    sharpe_ratios = sharpe_series.values

    # ORIGINAL-style prints (kept)
    if print_out:
        print("Mean Returns (r_i):", r_i)                 # np.array
        print("Volatilities (σ_p):", sigma_p)             # np.array
        print("Sharpe Ratios:", sharpe_ratios)           # np.array

        # NEW: per-asset mappings (ticker → value) with full precision
        print("r_i by asset:", r_i_series.to_dict())
        print("sigma_p by asset:", sigma_p_series.to_dict())
        print("Sharpe by asset:", sharpe_series.to_dict())

        # Top-N tickers (drop NaNs, sort desc)
        top_assets = sharpe_series.dropna().sort_values(ascending=False).head(top_n).index.tolist()
        print(f"Top {top_n} assets by Sharpe:", top_assets)

    # Return both the vector and the convenient Series + top list
    return {
        "returns_df": returns,               # for downstream use if needed
        "r_i": r_i,                          # np.array (original)
        "sigma": sigma,                      # covariance matrix (np.array, original)
        "sigma_p": sigma_p,                  # per-asset std as np.array (original-style)
        "sharpe": sharpe_ratios,             # np.array (original-style)
        "r_i_series": r_i_series,            # labeled Series
        "sigma_p_series": sigma_p_series,    # labeled Series
        "sharpe_series": sharpe_series,      # labeled Series
        "top_assets": sharpe_series.dropna().sort_values(ascending=False).head(top_n).index.tolist()
    }

def load_data(path=None):
        """
        Load price data from a CSV file.

        Parameters:
        - path (str, optional): Specific file path to load. If not provided,
            the method searches for the most recent 'all_data_*.csv' in the current directory.

        Returns:
        - pd.DataFrame: Loaded price data with datetime index, or None if nothing is found.
        """
        if path and os.path.exists(path):
            print(f"Loading data from: {path}")
            data = pd.read_csv(path, index_col=0, parse_dates=True)
            return data

        pattern = os.path.join(".", "all_data_*.csv")
        files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        if files:
            print(f"Loading latest data from: {files[0]}")
            data = pd.read_csv(files[0], index_col=0, parse_dates=True)
            return data

        print("No CSV file found in current directory.")
        return None
