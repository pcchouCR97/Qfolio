import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

class DailyBenchmarkSimulator:
    """
    Simulates a buy-and-hold benchmark strategy with daily value tracking.

    Key differences from BenchmarkSimulator:
    1. Returns daily values, not just rebalance-date values
    2. Adds new investment only on specified rebalance dates
    3. Uses same date alignment as the optimized portfolio for fair comparison

    This ensures apples-to-apples comparison between the quantum-optimized
    portfolio and the passive benchmark.
    """

    def __init__(self, assets, data, init_budget, new_investment):
        """
        Initialize the daily benchmark simulator.

        Args:
            assets (list): List of asset tickers to track (e.g., ['VOO'])
            data (pd.DataFrame): Price data with tickers as columns, dates as index
            init_budget (float): Initial investment amount
            new_investment (float): Amount to invest at each rebalance date
        """
        self.assets = assets
        self.data = data
        self.init_budget = init_budget
        self.new_investment = new_investment

    def simulate(self, start_date, end_date, rebalance_dates=None):
        """
        Simulate buy-and-hold strategy with daily tracking.

        Args:
            start_date (str): Simulation start date (YYYY-MM-DD)
            end_date (str): Simulation end date (YYYY-MM-DD)
            rebalance_dates (pd.DatetimeIndex, optional): Specific dates when new
                investment is added. If None, no additional investment is made.

        Returns:
            tuple: (daily_values_df, daily_roi_df)
                - daily_values_df: DataFrame with daily portfolio values for each asset
                - daily_roi_df: DataFrame with daily ROI (%) for each asset
        """
        # Convert dates to timestamps
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)

        # Adjust start date to first available trading day
        start_ts_adj = start_ts
        while start_ts_adj not in self.data.index:
            start_ts_adj += BDay(1)
            if start_ts_adj > end_ts:
                raise ValueError(f"No valid trading days between {start_date} and {end_date}")

        # Get all trading days in the simulation period
        simulation_dates = self.data.loc[start_ts_adj:end_ts].index

        if len(simulation_dates) == 0:
            raise ValueError(f"No data available between {start_date} and {end_date}")

        # Convert rebalance_dates to set for fast lookup
        if rebalance_dates is not None:
            rebalance_dates_set = set(pd.to_datetime(rebalance_dates))
        else:
            rebalance_dates_set = set()

        # Initialize tracking dictionaries for each asset
        daily_values = {asset: [] for asset in self.assets}
        daily_shares = {asset: [] for asset in self.assets}
        daily_invested = {asset: [] for asset in self.assets}

        # Track cumulative investment for ROI calculation
        cumulative_investment = {asset: self.init_budget for asset in self.assets}

        # Day 1: Initial purchase
        first_date = simulation_dates[0]
        for asset in self.assets:
            initial_price = self.data[asset].loc[first_date]
            initial_shares = self.init_budget / initial_price

            daily_shares[asset].append(initial_shares)
            daily_values[asset].append(self.init_budget)
            daily_invested[asset].append(self.init_budget)

        # Simulate remaining days
        for i, current_date in enumerate(simulation_dates[1:], start=1):
            for asset in self.assets:
                # Get current price
                current_price = self.data[asset].loc[current_date]

                # Check if this is a rebalance date (add new investment)
                if current_date in rebalance_dates_set:
                    # Add new investment: buy more shares
                    prev_shares = daily_shares[asset][-1]
                    new_shares = self.new_investment / current_price
                    total_shares = prev_shares + new_shares

                    # Update cumulative investment
                    cumulative_investment[asset] += self.new_investment
                else:
                    # No new investment, keep same shares
                    total_shares = daily_shares[asset][-1]

                # Calculate portfolio value
                portfolio_value = total_shares * current_price

                # Record daily data
                daily_shares[asset].append(total_shares)
                daily_values[asset].append(portfolio_value)
                daily_invested[asset].append(cumulative_investment[asset])

        # Convert to DataFrames
        daily_values_df = pd.DataFrame(daily_values, index=simulation_dates)
        daily_invested_df = pd.DataFrame(daily_invested, index=simulation_dates)

        # Calculate ROI: (portfolio_value - invested_capital) / invested_capital * 100
        daily_roi_df = ((daily_values_df - daily_invested_df) / daily_invested_df) * 100

        return daily_values_df, daily_roi_df

    def get_statistics(self, daily_values, daily_roi):
        """
        Calculate performance statistics for the benchmark.

        Args:
            daily_values (pd.DataFrame): Daily portfolio values
            daily_roi (pd.DataFrame): Daily ROI percentages

        Returns:
            dict: Dictionary of performance metrics for each asset
        """
        stats = {}

        for asset in self.assets:
            values = daily_values[asset]
            roi_series = daily_roi[asset]

            # Calculate daily returns
            daily_returns = values.pct_change().dropna()

            # Annualized metrics
            trading_days_per_year = 252
            total_days = len(values)

            # Total return
            total_return = roi_series.iloc[-1]

            # Annualized return
            years = total_days / trading_days_per_year
            annualized_return = ((1 + total_return/100) ** (1/years) - 1) * 100 if years > 0 else 0

            # Volatility (annualized)
            volatility = daily_returns.std() * np.sqrt(trading_days_per_year) * 100

            # Sharpe ratio (assuming 0% risk-free rate)
            sharpe = (annualized_return / volatility) if volatility > 0 else 0

            # Max drawdown
            cumulative_returns = (1 + daily_returns).cumprod()
            running_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min() * 100

            # Win rate (days with positive returns)
            win_rate = (daily_returns > 0).sum() / len(daily_returns) * 100 if len(daily_returns) > 0 else 0

            stats[asset] = {
                'Total Return (%)': total_return,
                'Annualized Return (%)': annualized_return,
                'Volatility (%)': volatility,
                'Sharpe Ratio': sharpe,
                'Max Drawdown (%)': max_drawdown,
                'Win Rate (%)': win_rate,
                'Total Days': total_days
            }

        return stats


# ==== Unit Test ==== #

def test_daily_benchmark_simulator():
    """
    Test the DailyBenchmarkSimulator with sample data.
    """
    print("=== Testing DailyBenchmarkSimulator ===\n")

    # Create sample price data
    dates = pd.date_range(start='2025-01-01', end='2025-01-31', freq='B')
    # Simulate VOO growing from $450 to $460 over the month
    voo_prices = np.linspace(450, 460, len(dates)) + np.random.randn(len(dates)) * 2

    sample_data = pd.DataFrame({'VOO': voo_prices}, index=dates)

    # Create simulator
    sim = DailyBenchmarkSimulator(
        assets=['VOO'],
        data=sample_data,
        init_budget=10000,
        new_investment=500
    )

    # Define rebalance dates (e.g., every 5 business days)
    rebalance_dates = dates[::5]

    # Run simulation
    daily_values, daily_roi = sim.simulate(
        start_date='2025-01-01',
        end_date='2025-01-31',
        rebalance_dates=rebalance_dates
    )

    print("Daily Values (first 10 days):")
    print(daily_values.head(10).round(2))
    print(f"\nDaily ROI (first 10 days):")
    print(daily_roi.head(10).round(2))

    # Get statistics
    stats = sim.get_statistics(daily_values, daily_roi)
    print(f"\n=== Performance Statistics ===")
    for asset, metrics in stats.items():
        print(f"\n{asset}:")
        for metric_name, value in metrics.items():
            if 'Ratio' in metric_name or 'Days' in metric_name:
                print(f"  {metric_name}: {value:.2f}")
            else:
                print(f"  {metric_name}: {value:.2f}%")

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    test_daily_benchmark_simulator()
