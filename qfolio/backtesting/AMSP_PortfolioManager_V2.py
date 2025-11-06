import pandas
from qfolio.backtesting.BenchmarkSimulator import BenchmarkSimulator
from qfolio.optimization.AMSP_PortfolioOptimizer import PortfolioOptimizer 

class PortfolioManager:
    def __init__(self,
                data: pandas.DataFrame,
                budget: float,
                new_invest: float,
                assets_portfolio: list,
                assets_benchmark: list,
                screener=None):
        self.data = data
        self.budget = budget
        self.new_invest = new_invest
        self.assets_portfolio = assets_portfolio
        self.assets_benchmark = assets_benchmark
        self.screener = screener
        self.optimizer = None # To hold the optimizer instance

    def monitor_and_select_assets(self, train_start, train_end, risk_free=0.0, top_n=3):
        """
        #Uses the screener to get the new top assets for the upcoming period.
        """
        if self.screener:
            print(f"Screening for top assets between {train_start} and {train_end}...")
            sharpe_results = self.screener(self.data, train_start, train_end, risk_free, top_n=len(self.data.columns)) # Get all assets

            # Filter for positive mean returns first
            positive_mean_returns = sharpe_results['r_i_series'][sharpe_results['r_i_series'] > 0]
            
            # Then, from these, select top assets by Sharpe Ratio
            if not positive_mean_returns.empty:
                candidate_sharpes = sharpe_results['sharpe_series'][positive_mean_returns.index]
                self.assets_portfolio = candidate_sharpes.dropna().sort_values(ascending=False).head(top_n).index.tolist()
            else:
                self.assets_portfolio = [] # No assets with positive mean return

            print(f"New top assets selected: {self.assets_portfolio}")
            return sharpe_results # Return full results

        print("No screener provided. Using initial assets.")
        return None # Return None if no screener

    def run_single_optimization(self, current_date, train_start, train_end, budget, k, lambda1, q, H_scale, solver_type, latest_open_prices):
        """
        Runs the optimization for a single period based on the provided training window.
        """
        print(f"--- Running optimization for period starting {current_date} ---")

        # Import the new optimizer class
        from AMSP_PortfolioOptimizerV2 import PortfolioOptimization_v2_3_AMSP

        # Initialize the optimizer with the current state, filtered by selected assets
        filtered_data_for_optimizer = self.data[self.assets_portfolio] 
        print(f"Assets passed to optimizer: {self.assets_portfolio}")
        
        self.optimizer = PortfolioOptimization_v2_3_AMSP(
            data=filtered_data_for_optimizer,
            k=k,
            lambda1=lambda1,
            q=q,
            H_scale=H_scale,
            solver_type=solver_type,
        )

        # Run the single-period optimization
        opt_result = self.optimizer.optimize_single_period(
            assets=self.assets_portfolio,
            budget=budget,
            current_date=current_date,
            train_start=train_start,
            train_end=train_end,
            latest_prices_dict=latest_open_prices
        )

        return opt_result

    def run_benchmark_simulator(self, start, end, freq, save, save_path):
        """
        # This function remains the same for comparing against buy-and-hold benchmarks.
        """
        self.benchmark = BenchmarkSimulator(
            assets=self.assets_benchmark + self.assets_portfolio,
            data=self.data,
            init_budget=self.budget,
            new_investment=self.new_invest
        )
        return self.benchmark.simulate(start, end, freq, save, save_path)
    
    