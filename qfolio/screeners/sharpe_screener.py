import numpy as np
import pandas as pd
import glob
import os
from qfolio.utils.utilities import StatisticCalculatorRollingD

def SharpeRatioCalculator(data, train_start, train_end, risk_free=0.0, top_n=3, print_out=False):
    """
    Summary: Compute per-asset Sharpe ratios over [train_start, train_end].

    Args:
        data (pd.DataFrame): Price data with DatetimeIndex
        train_start (str or pd.Timestamp): Start date for training period
        train_end (str or pd.Timestamp): End date for training period
        risk_free (float): Risk-free rate matching return interval (default 0.0)
        top_n (int): Number of top assets to identify by Sharpe ratio (default 3)
        print_out (bool): Whether to print detailed output (default False)
    
    Returns:
        dict: Contains original outputs and additional per-asset metrics:
            - "returns_df": pd.DataFrame of returns
            - "r_i": np.array of mean returns
            - "sigma": np.array covariance matrix
            - "sigma_p": np.array of per-asset std dev
            - "sharpe": np.array of Sharpe ratios
            - "r_i_series": pd.Series of mean returns (labeled)
            - "sigma_p_series": pd.Series of std dev (labeled)
            - "sharpe_series": pd.Series of Sharpe ratios (labeled)
            - "top_assets": list of top-N asset tickers by Sharpe ratio

    Example: 
        >>> sharpe_results = SharpeRatioCalculator(data, '2024-01-01', '2024-12-31', risk_free=0.01, top_n=5, print_out=True)
        >>> print(sharpe_results["top_assets"]) returns the top n=3 assets by Sharpe ratio, ['VOO', 'PLTR', 'AAPL'] for example.
    """
    # quick check to ensure top_n is not greater than number of assets
    if top_n > len(data.columns):
        raise ValueError(f"top_n ({top_n}) cannot be greater than the number of assets ({len(data.columns)})")

    # Calculate mean return using the default StatisticCalculatorRollingD function.
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
    # https://www.investopedia.com/terms/s/sharperatio.asp
    # https://en.wikipedia.org/wiki/Sharpe_ratio
    sharpe_series = (r_i_series - risk_free) / sigma_p_series.replace(0.0, np.nan)
    sharpe_ratios = sharpe_series.values

    if print_out:
        print("Mean Returns (r_i):", r_i)                # np.array
        print("Volatilities (sigma_p):", sigma_p)        # np.array
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

def load_data(path="qfolio/data/all_stocks_adjusted_close_t.csv"):
        """
        Summary: Load price data from a CSV file.

        Args:
            path (str, optional): Specific file path to load. If not provided, the method will use path = "qfolio/data/all_stocks_adjusted_close_t.csv".

        Returns:
            pd.DataFrame: Loaded price data with datetime index, or None if nothing is found.
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

# Test / example in the script itself
if __name__ == "__main__":
    data = load_data(path="qfolio/data/all_stocks_adjusted_close_t.csv")
    sharpe = SharpeRatioCalculator(data, train_start="2025-08-21", train_end="2025-09-21", risk_free=0.0, top_n=6, print_out=True)
    print("="*20)
    print(sharpe["top_assets"])
    print("Type of the return ['top_assets']:",type(sharpe["top_assets"]))
    print("="*20)







