import pandas as pd
import numpy as np
from scipy.stats import norm

class RiskMetricsCalculator:
    """
    A class to calculate portfolio risk metrics like VaR and CVaR using various methods.
    """
    def __init__(self, portfolio_allocation: dict, price_data: pd.DataFrame, current_date: pd.Timestamp, lookback_period: int = 200):
        """
        Initializes the RiskMetricsCalculator.

        Args:
            portfolio_allocation (dict): A dictionary mapping tickers to the number of shares.
            price_data (pd.DataFrame): A DataFrame with dates as the index and ticker prices as columns.
            current_date (pd.Timestamp): The current date in the simulation, used as the end of the lookback window.
            lookback_period (int): The number of trading days to look back for historical calculations.
        """
        if not portfolio_allocation or pd.Series(portfolio_allocation).empty:
            raise ValueError("Portfolio allocation cannot be empty.")
            
        self.portfolio_allocation = pd.Series(portfolio_allocation)
        self.price_data = price_data
        self.current_date = current_date
        self.lookback_period = lookback_period
        self.portfolio_returns = None # To cache portfolio returns
        
        self._prepare_data()

    def _prepare_data(self):
        """
        Prepares the historical price data by filtering for relevant tickers and dates.
        Handles assets with short histories by adjusting the lookback period.
        """
        tickers = self.portfolio_allocation.index.tolist()
        
        # 1. Filter price data for relevant tickers AND up to the current simulation date
        relevant_price_data = self.price_data.loc[:self.current_date, tickers].dropna(axis=0, how='all')

        # 2. Determine the valid date range for calculation
        # Find the first valid date where all assets have price data
        first_valid_date_for_all_assets = relevant_price_data.dropna().index.min()
        
        # Set the end date for our analysis
        end_date = self.current_date
        
        # Determine the start date based on the lookback period
        lookback_start_date = end_date - pd.tseries.offsets.BDay(self.lookback_period - 1)
        
        # The actual start date is the more recent of the two
        self.effective_start_date = max(first_valid_date_for_all_assets, lookback_start_date)
        self.effective_end_date = end_date
        
        # 3. Finalize the data slice for calculations
        self.final_data = relevant_price_data.loc[self.effective_start_date:self.effective_end_date]

    def get_portfolio_daily_returns(self) -> pd.Series:
        """
        Calculates the historical daily percentage returns of the entire portfolio.
        The result is cached to avoid redundant calculations.
        """
        if self.portfolio_returns is not None:
            return self.portfolio_returns

        # Calculate the total dollar value of the portfolio for each day
        portfolio_values = self.final_data.dot(self.portfolio_allocation)
        
        # Calculate the daily percentage change and drop the first NaN value
        daily_returns = portfolio_values.pct_change().dropna()
        
        self.portfolio_returns = daily_returns
        return self.portfolio_returns

    def calculate_portfolio_metrics(self, confidence_level: float = 0.95, method: str = 'historical') -> dict:
        """
        Calculates the VaR and CVaR for the entire portfolio.

        Args:
            confidence_level (float): The confidence level for the calculation (e.g., 0.95 for 95%).
            method (str): The calculation method ('historical' or 'parametric').

        Returns:
            A dictionary containing the calculated 'VaR' and 'CVaR'.
        """
        returns = self.get_portfolio_daily_returns()
        alpha = 1 - confidence_level

        if method == 'historical':
            # Historical VaR is the quantile of the historical returns
            var = returns.quantile(alpha)
            # Historical CVaR is the average of the returns that are worse than the VaR
            cvar = returns[returns <= var].mean()
            
        elif method == 'parametric':
            mean = returns.mean()
            std_dev = returns.std()
            
            # Parametric VaR based on the normal distribution
            z_score = norm.ppf(alpha)
            var = mean + z_score * std_dev
            
            # Parametric CVaR for a normal distribution
            cvar = mean - (std_dev / alpha) * norm.pdf(z_score)
            
        else:
            raise ValueError("Invalid method. Choose 'historical' or 'parametric'.")

        return {'VaR': var, 'CVaR': cvar}

    def calculate_individual_asset_metrics(self, confidence_level: float = 0.95, method: str = 'historical') -> pd.DataFrame:
        """
        Calculates the VaR and CVaR for each individual asset in the portfolio.

        Args:
            confidence_level (float): The confidence level for the calculation.
            method (str): The calculation method ('historical' or 'parametric').

        Returns:
            A DataFrame with each asset's VaR and CVaR.
        """
        results = {}
        alpha = 1 - confidence_level

        for ticker in self.final_data.columns:
            asset_returns = self.final_data[ticker].pct_change().dropna()
            
            if method == 'historical':
                var = asset_returns.quantile(alpha)
                cvar = asset_returns[asset_returns <= var].mean()
            
            elif method == 'parametric':
                mean = asset_returns.mean()
                std_dev = asset_returns.std()
                z_score = norm.ppf(alpha)
                var = mean + z_score * std_dev
                cvar = mean - (std_dev / alpha) * norm.pdf(z_score)
            
            else:
                raise ValueError("Invalid method. Choose 'historical' or 'parametric'.")
            
            results[ticker] = {'VaR': var, 'CVaR': cvar}
            
        return pd.DataFrame.from_dict(results, orient='index')

