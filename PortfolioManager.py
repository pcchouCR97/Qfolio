import pandas
from BenchmarkSimulator import BenchmarkSimulator
from PortfolioOptimizer import PortfolioOptimizer

class PortfolioManager:
    def __init__(self, 
                 data : pandas.DataFrame,
                 budget : float, 
                 new_invest: float,
                 assets_portfolio: list,
                 assets_benchmark: list):
        self.data = data
        self.budget = budget
        self.new_invest = new_invest
        self.assets_portfolio = assets_portfolio # assets that you want to have in your portfolio
        self.assets_benchmark = assets_benchmark # assets you want to compare with 

    def run_benchmark_simulator(self, start, end, freq, save, save_path):
        """
        Return df 
        """
        self.benchmark = BenchmarkSimulator(
            assets = self.assets_benchmark + self.assets_portfolio,
            data = self.data,
            init_budget = self.budget,
            new_investment = self.new_invest
        )
        return self.benchmark.simulate(start, end, freq, save, save_path)

    def run_portfolio_optimization(self, start, end, save, save_path, custom_prefix,
                                   k, lambda1, q, freq, t_freq, H_scale, solver_type):
        # build QP and solve using classical or quantum algorithm
        self.optimzation = PortfolioOptimizer(
            assets = self.assets_portfolio,
            data = self.data,
            budget = self.budget,
            new_investment = self.new_invest,
            k = k,
            lambda1 = lambda1,
            q = q,
            freq = freq,
            t_freq = t_freq,
            solver_type = solver_type,
            H_scale = H_scale
        )
        return self.optimzation.optimize(start, end, save, save_path, custom_prefix)
        
        