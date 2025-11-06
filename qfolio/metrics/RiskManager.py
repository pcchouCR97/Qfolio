import pandas as pd
import numpy as np

class RiskManager:
    def __init__(self, lookback_days=60, std_dev_threshold=1.0, sharpe_n=3, wait_days=5):
        self.lookback_days = lookback_days
        self.std_dev_threshold = std_dev_threshold
        self.sharpe_n = sharpe_n
        self.wait_days = wait_days
        self.daily_returns = pd.Series(dtype=float)
        self.cash_out_day = None

    def add_daily_return(self, date, portfolio_value, last_day_value):
        if last_day_value != 0:
            daily_return = (portfolio_value - last_day_value) / last_day_value
            self.daily_returns.at[date] = daily_return

    def check_risk_condition(self, current_date):
        if len(self.daily_returns) < self.lookback_days:
            return False

        # Get the returns for the lookback window
        window_returns = self.daily_returns.last(f'{self.lookback_days}D')

        if window_returns.empty:
            return False

        mean_return = window_returns.mean()
        std_dev = window_returns.std()

        print(f"Mean Return: {mean_return}, Std Dev: {std_dev}")

        if std_dev == 0:
            return False

        # Get the latest daily return
        latest_return = self.daily_returns.iloc[-1]

        # Check if the latest return is below the threshold
        if latest_return < (mean_return - self.std_dev_threshold * std_dev):
            return True

        return False

    def manage_risk(self, data, current_date, screener):
        # This method will contain the logic for cashing out and rebalancing
        # For now, it's a placeholder
        print("Risk condition met. Executing risk management.")

        # Calculate sharpe ratios for all assets
        sharpe_results = screener(data, 
                                  (current_date - pd.DateOffset(days=self.lookback_days)).strftime('%Y-%m-%d'), 
                                  current_date.strftime('%Y-%m-%d'),
                                  top_n=len(data.columns))

        # Check if all sharpe ratios are negative
        if all(sharpe_results['sharpe_series'] < 0):
            print("All Sharpe ratios are negative. Cashing out.")
            self.cash_out_day = current_date
            return "cash_out"

        return "hold"

    def post_cash_out_logic(self, data, current_date, screener):
        if self.cash_out_day is None:
            return "hold"

        days_since_cash_out = (current_date - self.cash_out_day).days
        if days_since_cash_out >= self.wait_days:
            print(f"{self.wait_days} day waiting period is over. Checking for rebalance opportunity.")
            # Calculate sharpe ratios for all assets
            sharpe_results = screener(data, 
                                      (current_date - pd.DateOffset(days=self.lookback_days)).strftime('%Y-%m-%d'), 
                                      current_date.strftime('%Y-%m-%d'),
                                      top_n=len(data.columns))
            
            voo_sharpe = sharpe_results['sharpe_series'].get('VOO', -np.inf)
            print(f"VOO Sharpe Ratio: {voo_sharpe}")

            strong_assets = sharpe_results['sharpe_series'][(sharpe_results['sharpe_series'] > 0) & (sharpe_results['sharpe_series'] > voo_sharpe)]

            if len(strong_assets) >= self.sharpe_n:
                print(f"Found {len(strong_assets)} assets with Sharpe ratio > VOO's Sharpe. Triggering rebalance.")
                self.cash_out_day = None # Reset cash out day
                return "rebalance"
            else:
                print("Not enough strong assets to rebalance. Waiting.")
                return "wait"
        else:
            print(f"In cash out waiting period. Day {days_since_cash_out+1}/{self.wait_days}")
            return "wait"


class RiskManagerV2:
    def __init__(self, drop_threshold=0.03, sharpe_n=3, rebalance_freq='10B', filter_by_mean_return=False):
        self.drop_threshold = drop_threshold
        self.sharpe_n = sharpe_n
        self.rebalance_freq = rebalance_freq
        self.triggered = False
        self.daily_returns = pd.Series(dtype=float)
        self.filter_by_mean_return = filter_by_mean_return

    def add_daily_return(self, date, portfolio_value, last_day_value):
        if last_day_value != 0:
            daily_return = (portfolio_value - last_day_value) / last_day_value
            self.daily_returns.at[date] = daily_return

    def check_risk_condition(self, current_date):
        if len(self.daily_returns) < 1:
            return False
            
        latest_return = self.daily_returns.iloc[-1]
        if latest_return < -self.drop_threshold:
            self.triggered = True
            self.trigger_date = current_date
            return True
        return False

    def manage_risk(self, data, current_date, screener, btrck_ref_dates = 60):
        print(f"--- Risk condition met on {current_date.strftime('%Y-%m-%d')} (drop > 3%). Executing risk management.")
        
        # Check for assets with positive sharpe ratio
        sharpe_results = screener(data, 
                                  (current_date - pd.DateOffset(days=btrck_ref_dates)).strftime('%Y-%m-%d'), 
                                  current_date.strftime('%Y-%m-%d'),
                                  top_n=len(data.columns))
        
        # Filter assets based on the new strategy
        if self.filter_by_mean_return:
            # First, filter for positive mean returns
            filtered_by_mean_return = sharpe_results['r_i_series'][sharpe_results['r_i_series'] > 0]
            # Then, from these, consider their Sharpe ratios
            candidate_sharpes = sharpe_results['sharpe_series'][filtered_by_mean_return.index]
            strong_assets = candidate_sharpes.dropna().sort_values(ascending=False) # Sort by Sharpe
        else:
            # Original logic: filter by positive Sharpe ratio
            strong_assets = sharpe_results['sharpe_series'][sharpe_results['sharpe_series'] > 0].dropna().sort_values(ascending=False)

        if len(strong_assets) >= self.sharpe_n:
            print(f"Found {len(strong_assets)} assets meeting criteria. Triggering immediate rebalance.")
            return "rebalance"
        else:
            print("Not enough assets meeting criteria. Cashing out and waiting for next rebalance day.")
            return "cash_out"

    def post_cash_out_logic(self, data, current_date, screener, actual_rebalance_dates, high_stopper_triggered=False):
        if not self.triggered:
            return "hold"

        # Daily check for rebalance opportunity while in cash
        print(f"Daily check for rebalance opportunity while in cash on {current_date.strftime('%Y-%m-%d')}.")
        
        sharpe_results = screener(data, 
                                  (current_date - pd.DateOffset(days=60)).strftime('%Y-%m-%d'), 
                                  current_date.strftime('%Y-%m-%d'),
                                  top_n=len(data.columns))
        
        voo_sharpe = sharpe_results['sharpe_series'].get('VOO', -np.inf)
        print(f"VOO Sharpe Ratio: {voo_sharpe}")

        # Filter assets based on the new strategy
        if self.filter_by_mean_return:
            # First, filter for positive mean returns
            filtered_by_mean_return = sharpe_results['r_i_series'][sharpe_results['r_i_series'] > 0]
            # Then, from these, consider their Sharpe ratios that also beat VOO
            candidate_sharpes = sharpe_results['sharpe_series'][filtered_by_mean_return.index]
            strong_assets = candidate_sharpes[(candidate_sharpes > voo_sharpe)].dropna().sort_values(ascending=False)
        else:
            # Original logic: filter by positive Sharpe ratio and beating VOO
            strong_assets = sharpe_results['sharpe_series'][(sharpe_results['sharpe_series'] > voo_sharpe) & (sharpe_results['sharpe_series'] > 0)].dropna().sort_values(ascending=False)

        if high_stopper_triggered:
            if len(strong_assets) > 0: # At least one strong asset found
                print(f"High Stopper Re-entry: Found exactly {self.sharpe_n} strong assets. Triggering rebalance.")
                self.triggered = False
                self.trigger_date = None
                # Reset high_stopper_triggered in AMSP_test_bed.py after rebalance
                return "rebalance"
            else:
                print(f"High Stopper Re-entry: No strong assets found. Holding cash.") 
                return "wait"
        else: # Original risk manager re-entry logic
            if len(strong_assets) >= self.sharpe_n:
                print(f"Found {len(strong_assets)} strong assets. Triggering rebalance.")
                self.triggered = False
                self.trigger_date = None
                return "rebalance"
            else:
                print("Not enough strong assets to rebalance. Holding cash.")
                return "wait"
