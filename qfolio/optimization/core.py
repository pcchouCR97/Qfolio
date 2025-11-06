import pandas
from qfolio.metrics.RiskMetrics import RiskMetricsCalculator
from qfolio.backtesting.PortfolioManager_AMSP import PortfolioManager
from qfolio.backtesting.portfolio import calculate_sector_concentration, calculate_sector_exposure
from qfolio.data.assets_loader import get_eligible_stocks
from qfolio.screeners.sharpe_screener import SharpeRatioCalculator
from qfolio.utils.debug_print_dev import debug_print
from dataclasses import dataclass
from typing import Optional

@dataclass
class OptimizationConfig:
    """Configuration for optimization parameters"""
    k: int
    lambda1: float
    q: float
    H_scale: float
    solver_type: str
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'OptimizationConfig':
        """Create config from dictionary"""
        return cls(**config_dict)
    
    @classmethod
    def default(cls) -> 'OptimizationConfig':
        """Return default configuration"""
        return cls(
            k=2,
            lambda1=1000.0,
            q=0.001,
            H_scale=1,
            solver_type='classic'
        )    

class BackTrackCore:

    def __init__(self, 
                 data, 
                 budget, 
                 sp_n,
                 actual_rebalance_dates, 
                 risk_lookback_period, 
                 initial_eligible_stocks,
                 simulation_days,
                 PortfolioManager,
                 new_invest_per_period,
                 all_trading_days,
                 training_lookback_days,
                 trading_universe_full,
                 opt_config: Optional[OptimizationConfig] = None):  
        self.data = data
        self.budget = budget
        self.sp_n = sp_n
        self.actual_rebalance_dates = actual_rebalance_dates
        self.risk_lookback_period = risk_lookback_period
        self.initial_eligible_stocks = initial_eligible_stocks
        self.simulation_days = simulation_days
        self.PM = PortfolioManager
        self.new_invest_per_period = new_invest_per_period
        self.all_trading_days = all_trading_days
        self.training_lookback_days = training_lookback_days
        self.trading_universe_full = trading_universe_full
        self.opt_config = opt_config or OptimizationConfig.default()


    def run(self, advanced = False, Risk = False, cvar_confidence_level = 0.95):
        # 3. --- Main Simulation Loop (Daily) ---
        portfolio_history = []
        current_budget = self.budget
        last_allocation = {}
        portfolio_high_value = self.budget
        all_rebalance_events = []
        is_first_rebalance = True  # Track first rebalance to exclude new investment
        previous_eligible_stocks = self.initial_eligible_stocks  # Track universe changes

        # Initialize advanced tracking if needed
        if advanced:
            # lists of advanced analysis
            composition_history = []  # Track which stocks selected at each rebalance
            turnover_history = []  # Track portfolio turnover at each rebalance
            previous_allocation = {}  # For turnover calculation
            sector_exposure_history = []  # Track sector weights at each rebalance
            concentration_history = []  # Track sector concentration (HHI) over time

        if Risk:
            # --- CVaR Risk Monitoring (For Analysis Only - Does Not Trigger Trades) ---
            if last_allocation:
                try:
                    risk_calc = RiskMetricsCalculator(
                        portfolio_allocation=last_allocation,
                        price_data = self.data,
                        current_date = current_date,
                        lookback_period = self.risk_lookback_period
                    )
                    portfolio_risk = risk_calc.calculate_portfolio_metrics(confidence_level=cvar_confidence_level, method='historical')
                    cvar_95 = portfolio_risk['CVaR']
                    var_95 = portfolio_risk['VaR']

                    # MONITORING ONLY - print if risk is elevated, but don't act on it
                    if cvar_95 < -0.10:  # Only warn if risk is significant
                        print(f"  [RISK MONITOR] {current_date.strftime('%Y-%m-%d')}: CVaR = {cvar_95:.2%}, VaR = {var_95:.2%}")
                except Exception as e:
                    pass  # Silent failure for monitoring

        for i, current_date in enumerate(self.simulation_days):
            # Calculate portfolio value for the current day
            if last_allocation and (current_date in self.data.index):
                prices_today = self.data.loc[current_date].to_dict()
                current_portfolio_value = sum(last_allocation.get(ticker, 0) * prices_today.get(ticker, 0) for ticker in last_allocation)
            else:
                current_portfolio_value = current_budget

            portfolio_high_value = max(portfolio_high_value, current_portfolio_value)
            portfolio_history.append({'date': current_date, 'value': current_portfolio_value, 'allocation': last_allocation})

            # Rebalance logic (only on scheduled dates)
            is_regular_rebalance = current_date in self.actual_rebalance_dates

            if is_regular_rebalance:
                all_rebalance_events.append(current_date)
                current_budget_for_opt = current_portfolio_value

                # Only add new investment after the first rebalance
                if not is_first_rebalance:
                    current_budget_for_opt += self.new_invest_per_period
                    print(f"--- Rebalancing on {current_date.strftime('%Y-%m-%d')} --- (Budget: ${current_portfolio_value:.2f} + New Investment: ${self.new_invest_per_period:.2f} = ${current_budget_for_opt:.2f})")
                else:
                    print(f"--- First Rebalancing on {current_date.strftime('%Y-%m-%d')} --- (Initial Budget: ${current_budget_for_opt:.2f})")
                    is_first_rebalance = False

                current_date_loc = self.all_trading_days.get_loc(current_date)
                train_end_loc = current_date_loc - 1
                if train_end_loc < 0:
                    print("Not enough historical data to train. Skipping rebalance.")
                    continue
                train_end = self.all_trading_days[train_end_loc]
                train_start_loc = train_end_loc - self.training_lookback_days
                if train_start_loc < 0: train_start_loc = 0
                train_start = self.all_trading_days[train_start_loc]

                # --- DYNAMIC UNIVERSE: Update eligible stocks based on data availability ---
                current_eligible_stocks = get_eligible_stocks(
                    current_date=train_end,
                    all_stocks=self.trading_universe_full,
                    data=self.data,
                    lookback_days=self.training_lookback_days + self.risk_lookback_period
                )

                #debug_print(current_eligible_stocks)

                # Check for universe expansion
                if len(current_eligible_stocks) > len(previous_eligible_stocks):
                    new_stocks = set(current_eligible_stocks) - set(previous_eligible_stocks)
                    print(f"  [UNIVERSE EXPANSION] +{len(new_stocks)} new stocks now eligible: {sorted(new_stocks)}, Total eligible stocks: {len(current_eligible_stocks)}")
                    previous_eligible_stocks = current_eligible_stocks

                # Screen assets from current eligible universe
                data_for_screening = self.data[current_eligible_stocks]
                current_sharpe_results = SharpeRatioCalculator(data_for_screening, train_start.strftime('%Y-%m-%d'),train_end.strftime('%Y-%m-%d'), risk_free=0.0, print_out=False)

                # Select top N assets from eligible universe
                positive_mean_returns = current_sharpe_results['r_i_series'][current_sharpe_results['r_i_series'] > 0]
                if not positive_mean_returns.empty:
                    candidate_sharpes = current_sharpe_results['sharpe_series'][positive_mean_returns.index]
                    selected_assets = candidate_sharpes.dropna().sort_values(ascending=False).head(self.sp_n).index.tolist()
                else:
                    selected_assets = []

                # Update PM with selected assets
                self.assets_portfolio = selected_assets

                print(f"Screening for top assets between {train_start.strftime('%Y-%m-%d')} and {train_end.strftime('%Y-%m-%d')}...")
                print(f"New top assets selected: {selected_assets}")

                if not self.assets_portfolio:
                    print("--- No assets with positive Sharpe. Maintaining previous allocation. ---")
                    if last_allocation:
                        # Keep previous allocation, just update value with current prices
                        current_portfolio_value = sum(
                            last_allocation.get(ticker, 0) * self.data.loc[current_date, ticker]
                            for ticker in last_allocation if ticker in self.data.columns
                        )
                        portfolio_history[-1]['value'] = current_portfolio_value
                    continue

                selected_assets_r_i = current_sharpe_results["r_i_series"].loc[self.assets_portfolio]
                if all(selected_assets_r_i < 0):
                    print("--- All selected assets have negative expected returns. Maintaining previous allocation. ---")
                    if last_allocation:
                        # Keep previous allocation, just update value with current prices
                        current_portfolio_value = sum(
                            last_allocation.get(ticker, 0) * self.data.loc[current_date, ticker]
                            for ticker in last_allocation if ticker in self.data.columns
                        )
                        portfolio_history[-1]['value'] = current_portfolio_value
                    continue
                
                # Update PM data with new eligible stocks
                self.PM.assets_portfolio = self.assets_portfolio  # Update the asset list
                self.PM.data = self.data[self.assets_portfolio]   # Update the data

                # Run optimization with current eligible stocks
                opt_result = self.PM.run_single_optimization(
                    current_date=current_date.strftime('%Y-%m-%d'),
                    train_start=train_start.strftime('%Y-%m-%d'),
                    train_end=train_end.strftime('%Y-%m-%d'),
                    budget=current_budget_for_opt,
                    k=self.opt_config.k, 
                    lambda1=self.opt_config.lambda1, 
                    q=self.opt_config.q, 
                    H_scale=self.opt_config.H_scale, 
                    solver_type=self.opt_config.solver_type, # or **{k: v for k, v in self.opt_config.__dict__.items()}
                    latest_open_prices=self.data.loc[current_date].to_dict()
                )
                #debug_print(opt_result)
                last_allocation = opt_result['allocation']
                portfolio_history[-1]['allocation'] = last_allocation
                portfolio_history[-1]['value'] = opt_result['value']
                portfolio_high_value = opt_result['value']

                """
                # Only for advanced analysis 
                # --- Track Portfolio Composition ---
                composition_history.append({
                    'date': current_date,
                    'assets': selected_assets.copy(),
                    'num_assets': len(last_allocation),
                    'allocation_dict': last_allocation.copy()
                })
                """

                if advanced:
                    # --- Calculate and Track Turnover ---
                    if previous_allocation:
                        # Calculate portfolio values
                        old_value = sum(previous_allocation.get(t, 0) * self.data.loc[current_date, t]
                                    for t in previous_allocation if t in self.data.columns)
                        new_value = sum(last_allocation.get(t, 0) * self.data.loc[current_date, t]
                                    for t in last_allocation if t in self.data.columns)

                        # Turnover = sum of absolute changes in weights
                        turnover = 0
                        all_tickers = set(list(previous_allocation.keys()) + list(last_allocation.keys()))
                        for ticker in all_tickers:
                            if ticker not in self.data.columns:
                                continue
                            old_weight = (previous_allocation.get(ticker, 0) * self.data.loc[current_date, ticker] / old_value) if old_value > 0 else 0
                            new_weight = (last_allocation.get(ticker, 0) * self.data.loc[current_date, ticker] / new_value) if new_value > 0 else 0
                            turnover += abs(new_weight - old_weight)

                        turnover_history.append({'date': current_date, 'turnover': turnover * 100})
                        print(f"  Portfolio Turnover: {turnover*100:.2f}%")

                    previous_allocation = last_allocation.copy()

                if advanced:
                    # --- Track Sector Exposure ---
                    if last_allocation:
                        sector_exp = calculate_sector_exposure(last_allocation, self.data.loc[current_date].to_dict())
                        sector_exposure_history.append({'date': current_date, **sector_exp})

                        # Calculate concentration
                        hhi = calculate_sector_concentration(sector_exp)
                        concentration_history.append({
                            'date': current_date,
                            'HHI': hhi,
                            'num_sectors': len(sector_exp)
                        })

                        print(f"  Sector Exposure: {', '.join([f'{s}: {w:.1f}%' for s, w in sorted(sector_exp.items(), key=lambda x: x[1], reverse=True)])}")
                        print(f"  Sector HHI: {hhi:.1f} ({len(sector_exp)} sectors)")

                print(f"End of Rebalance Period. Portfolio Value: ${opt_result['value']:.2f}")
        print("--- Simulation Complete ---")
        return portfolio_history, all_rebalance_events
