import numpy as np
import os
import pandas as pd
import sys
import time
from pandas.tseries.offsets import BDay, BMonthBegin
from pandas.tseries.frequencies import to_offset
from PortfolioOptimization import PortfolioOptimizationHamiltonian
from print_n_plots import decode_results_rolling
from utilities import Estimate_num_bits, StatisticCalculatorRolling, StatisticCalculatorRollingD
from StockDataCollector import StockDataCollector
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_algorithms.utils import algorithm_globals
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms import QAOA, SamplingVQE
from qiskit_aer.primitives import Sampler
from qiskit.circuit.library import TwoLocal

class Benchmark():
    def __init__(self,
                 data: pd.DataFrame, # All history database
                 benchmark_tickers: list, # Tickers that you want to compare with
                 portfolio_tickers: list,
                 budget: float, # Your budget
                 q:list,
                 new_investment_amount: float, # new investment amount 
                 lambda1 : list, # Penalty coefficient lambda 1, minimal risk
                 lambda2 : list, # Penalty coefficient lambda 2, budget constraint
                 expected_returns_rate: list, # Expected returns rate each trainning peroid
                 benchmark_start_date: str, # Benchmark start date
                 benchmark_end_date: str, # Benchmark end date
                 freq: str, # Split date frequency (period) at least 3 months
                 train_freq: list, # How many period you want
                 solver_type: str,
                 ) -> None:
        
        self.data = data
        self.bm_tickers = benchmark_tickers
        self.opt_tks = portfolio_tickers
        self.B = budget 
        # Ensure all are lists
        self.q = q if isinstance(q, list) else [q]
        self.new_invst = new_investment_amount
        self.lbda1 = lambda1 if isinstance(lambda1, list) else [lambda1]
        self.lbda2 = lambda2 if isinstance(lambda2, list) else [lambda2]
        self.exp_rr = expected_returns_rate if isinstance(expected_returns_rate, list) else [expected_returns_rate]
        self.bms = benchmark_start_date
        self.bme = benchmark_end_date
        self.freq = freq
        self.t_freq = train_freq if isinstance(train_freq, list) else [train_freq]
        self.solver_type = solver_type

        return None
    
    def RunBenchmarkBaseD(self, save_to_csv = False, csv_path = None, print_debug = False):
        """
        Run Benchmark for base tickers of your choice.

        Args: 
            1. save_to_csv: Boolean | Default = False | True if save file to desinated folder.
            2. print_debug: Boolean | Default = False | Print something for debugging purpose.

        Return: 
            1. df_benchmarks_returns: dict | Return a database for each tickers performance

        """
        # Ensure bms is adjusted to the first available market date
        bms_adj = pd.Timestamp(self.bms) # (b)ench(m)ark (s)tart date (adj)ust.
        while bms_adj not in self.data[self.bm_tickers].index:
            bms_adj += BDay(1)  # Move to the next business day if necessary
        bm_val = {ticker: [self.B] for ticker in self.bm_tickers}  

        # Buy initial shares at the first available market date
        bm_shares = {
            ticker: [self.B / self.data[ticker].loc[bms_adj]]
            for ticker in self.bm_tickers
            }
        # Step through every nth date using freq logic
        offset = to_offset(self.freq)
        step = offset.n if hasattr(offset, 'n') else 1

        # Get the list of all valid dates
        valid_dates = self.data.loc[self.bms:self.bme].index
        print(valid_dates)
        # Get crps (current rebalancing points) and nrps (next periods)
        crps = valid_dates[::step] # (c)urrent (r)eturn (p)eriod: current rebalance date
        nrps = valid_dates[step::step] # (n)ext (r)eturn (p)eriod: next rebalance date 
        print(step)
        print(crps)
        for crp, nrp in zip(crps, nrps):
            #print(f"current {crp}, next {nrp}")
            for ticker in self.bm_tickers:
                # Get last recorded value & shares
                last_value = bm_val[ticker][-1]
                last_shares = bm_shares[ticker][-1] # get 01/01 shares
                
                if crp != self.bms : 
                    benchmark_value = last_shares * self.data[ticker].loc[nrp] + self.new_invst
                    bm_val[ticker].append(benchmark_value)
                # Update new shares
                extra_shares = self.new_invst/self.data[ticker].loc[nrp]
                new_shares = last_shares + extra_shares
                #print(new_shares)
                bm_shares[ticker].append(new_shares) 

        # Convert benchmark values to DataFrame
        df_benchmarks = pd.DataFrame(bm_val, index=crps, columns=[ticker for ticker in self.bm_tickers])

        invested_amounts = [self.B + i * self.new_invst for i in range(len(df_benchmarks))]

        # Convert to Series aligned with index
        invested_series = pd.Series(invested_amounts, index=df_benchmarks.index)

        # Step 2: Broadcast to match all columns (tickers)
        invested_df = pd.DataFrame({col: invested_series for col in df_benchmarks.columns})

        # Cumulative return accounting for new capital
        df_benchmarks_returns = ((df_benchmarks - invested_df) / invested_df) * 100

        if save_to_csv:
            # Define the subdirectory path
            results_dir = csv_path

            # Ensure the directory exists
            os.makedirs(results_dir, exist_ok=True)

            # Save Database
            filename = f"benchmarks_returns_freq_{self.freq}.csv"
            df_benchmarks_returns.to_csv(os.path.join(results_dir, filename), index=False)
            
        return df_benchmarks_returns

    def RunBenchmarkOptimizedOptPortfolioD(self, save_to_csv = False, csv_path = None,custom_prefix=""):
        # Set seeds for reprodcibility
        seed = 123456

        data = self.data[self.opt_tks]
        # Get the last available stock prices (latest close prices)
        latest_prices = data.iloc[-1]  # Use last row to get most recent prices
        #print("Latest Stock Prices:\n", latest_prices)
        prices = latest_prices.values

        # Estimate how many bits we need for binary encoding
        num_bits = Estimate_num_bits(self.B, prices)
        execution_times = []
        for q in self.q: # qisk aversion coefficient
            for lambda1 in self.lbda1: # budget
                for lambda2 in self.lbda2: # return
                    for expected_returns_rate in self.exp_rr: # resonable period return rate
                        for t_freq in self.t_freq:
                            # Step through every nth date using freq logic
                            offset = to_offset(self.freq)
                            
                            step = offset.n if hasattr(offset, 'n') else 1

                            # Get the list of all valid dates
                            valid_dates = self.data.loc[self.bms:self.bme].index

                            # Get crps (current rebalancing points) and nrps (next periods)
                            crps = valid_dates[::step] # (c)urrent (r)eturn (p)eriod: current rebalance date
                            nrps = valid_dates[step::step] # (n)ext (r)eturn (p)eriod: next rebalance date 
                            
                            # Initialize list to store results for optimized portfolio
                            results_data = []
                            allocation_results = []
                            portfolio_value = []
                            return_ = []
                            budget = self.B
                            cost_diff = []
                            
                            # Makesure train_freq doesn't have to equals to freq
                            #print(t_freq)
                            train_offset = to_offset(t_freq)
                            train_step = train_offset.n if hasattr(train_offset, 'n') else 1
                            
                            # Start timer
                            start_time = time.time()
                            for i in range(len(crps)):
                                crp = crps[i]
                                nrp = nrps[i] if i < len(nrps) else None  # handle last crp without nrp

                                idx = data.index.get_loc(crp)
                                train_data = data.iloc[max(0, idx - train_step):idx]
                                
                                print(f" TRAINING START date: {train_data.index[0]}")
                                print(f" TRAINING END date: {train_data.index[-1]}")
                                print(f" (C)urrent (R)eturn (P)eroid: {crp}")
                                print(f" (N)ext (R)eturn (P)eroid {nrp}")
                                
                                returns, r_i, sigma = StatisticCalculatorRollingD(train_data, 
                                                                                train_start=train_data.index[0],
                                                                                train_end=train_data.index[-1],
                                                                                freq = t_freq, 
                                                                                print_out = False, 
                                                                                plot = False)
                                
                                # Get the closest past available trading day
                                closest_date = pd.Timestamp(crp)
                                if closest_date not in self.data[self.opt_tks].index:
                                    closest_date += BDay(1)  # Move to the next business day if necessary
                                
                                # Fetch prices for the closest available market date
                                latest_prices = np.array(data.loc[closest_date].values) # Current month lastest price
                                
                                # Constructing our objective function
                                portfoilo_Hamiltonian = PortfolioOptimizationHamiltonian(
                                    expected_returns = r_i,
                                    covariances = sigma,
                                    budget = budget,
                                    q = q,
                                    lambda1 = lambda1, # Penalty coefficient 1
                                    lambda2 = lambda2, # Penalty coefficient 2
                                    prices = latest_prices, # Current month lastest price
                                    max_binary_bits = num_bits,
                                    expected_returns_rate = expected_returns_rate # expected return rate each time period from budget
                                )
                                
                                #portfoilo_Hamiltonian
                                qp = portfoilo_Hamiltonian.to_quadratic_program() # return a quadratic program
                                #print(qp)
                                #sys.exit("Stopping execution here")
                                opt_results = self.solve_qp(qp, num_assets = len(r_i))
                                
                                # Store results in a dictionary
                                results_data.append({
                                    "Date": crp.strftime("%Y-%m-%d"),
                                    "Objective Value": opt_results.fval,
                                    "Optimal Selection": list(opt_results.x)
                                })

                                latest_prices_zip = dict(zip(self.opt_tks, latest_prices))

                                #print(f"before_optimized_current_value:{budget}")

                                share_allocation_classic = decode_results_rolling(result=opt_results, 
                                                                                tickers=self.opt_tks, 
                                                                                num_bits = num_bits, 
                                                                                crp=crp,
                                                                                latest_prices = latest_prices_zip)
                                allocation_results.append(share_allocation_classic)
                                val_crp = sum(share_allocation_classic[ticker] * latest_prices_zip[ticker] for ticker in data)
                                portfolio_value.append(val_crp) # portfolio value after applying allocation_results
                                cost_diff.append(val_crp - budget) # let's see how strong the budget constraint leads us!
                            
                                # MWR (next period)
                                # [portfolio_value[crp] - total invest (budget + new amount)]/ total invest (budget + new amount)
                                # Money-Weighted Return (MWR)
                                # simple total return with cash flows
                                if i==0: # this is the return on the rebalance date before adding extra fundings
                                    ret = (100*(val_crp - portfolio_value[-1])/portfolio_value[-1]) 
                                    return_.append(ret) # should always be 0

                                elif i !=0 :
                                    ret = (100*(portfolio_value[-1] - 
                                                (portfolio_value[-2] + self.new_invst))
                                                /(portfolio_value[-2] + self.new_invst))
                                    return_.append(ret) # this is the return on the rebalance date before adding extra fundings

                                # === Calculate our portfolio value for the next period when we use allocation_results
                                # The next month is the first invest month, therefore, we will start put our money in it.
                                if nrp:
                                    while nrp not in self.data[self.opt_tks].index:
                                        nrp += BDay(1)  # Move to the next business day if necessary
                                    new_latest_prices = np.array(data.loc[nrp].values)

                                # Next period ticker price
                                new_latest_prices_zip = dict(zip(self.opt_tks, new_latest_prices))

                                # Value on next month based on allocation from previous month at the sametime
                                val_nrp = sum(share_allocation_classic[ticker] * new_latest_prices_zip[ticker] for ticker in data)

                                # Update our budget
                                budget = val_nrp + self.new_invst
                                
                                print(f"==== this {crp} train loop end ====")
                                
                            # End timer
                            end_time = time.time()  
                            exec_time = end_time - start_time  # Calculate duration
                            
                            # Store execution time for analysis
                            execution_times.append([lambda1, lambda2, expected_returns_rate, exec_time])

                            print(f"----- Final Portfolio Share Allocation -----")
                            print(pd.DataFrame(allocation_results).set_index('Date'))
                            print(f"Value of the optimized portfoilo (cost of the last rebalance period): {(val_nrp)}")


                            # ===================== Post-Processing ===================== #

                            #print((portfolio_value))

                            #crp, nrp
                            df_optimized_portfolio = pd.DataFrame(portfolio_value, index=crps)

                            # Generate invested capital at each rebalance point
                            invested_amounts = [self.B + i * self.new_invst for i in range(len(df_optimized_portfolio))]

                            # Convert to Series aligned with index
                            invested_series = pd.Series(invested_amounts, index=df_optimized_portfolio.index)
                            #print(f"invested_series: {invested_series}")
                            # Cumulative return accounting for new capital
                            df_portfolio_returns = ((df_optimized_portfolio.squeeze() - invested_series) / invested_series) * 100

                            # ===================== Save dataframe ===================== #
                            if save_to_csv:
                                # set result directory
                                results_dir = csv_path   
                                # Save data
                                filename = f"{custom_prefix}port_return_q_{q}_lambda1_{lambda1}_lambda2_{lambda2}_exp_rr_{expected_returns_rate}_freq_{self.freq}_tfreq_{t_freq}.csv"
                                df_portfolio_returns.to_csv(os.path.join(results_dir, filename), index=False)

                                # Save difference data 
                                df_cost_diff = pd.DataFrame(cost_diff, index=crps)
                                filename = f"{custom_prefix}cost_diff_q_{q}_lambda1_{lambda1}_lambda2_{lambda2}_exp_rr_{expected_returns_rate}_freq_{self.freq}_tfreq_{t_freq}.csv"
                                df_cost_diff.to_csv(os.path.join(results_dir, filename), index=False)

                                # Save allocation data for each combination    
                                # diff(): computes change between rows
                                # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.diff.html#pandas.DataFrame.diff
                                # abs().gt(0): checks if there was a change
                                # sum(axis=1): counts how many assets changed = number of transactions
                                allocation_results = pd.DataFrame(allocation_results).set_index('Date')
                                allocation_results['Transcations'] = allocation_results.diff().abs().gt(0).sum(axis=1)

                                # 1. Sum all transaction values
                                total_transactions = allocation_results['Transcations'].sum()

                                # 2. Create a new row with NaNs for asset columns, and total in 'transaction'
                                summary_row = pd.DataFrame(
                                    [[None] * (allocation_results.shape[1] - 1) + [total_transactions]],
                                    columns=allocation_results.columns,
                                    index=[('Total Transcations')]
                                )

                                # 3. Append it to the DataFrame
                                allocation_results = pd.concat([allocation_results, summary_row])
                                filename = f"{custom_prefix}allocation_q_{q}_lambda1_{lambda1}_lambda2_{lambda2}_exp_rr_{expected_returns_rate}_freq_{self.freq}_tfreq_{t_freq}.csv"
                                allocation_results.to_csv(os.path.join(results_dir, filename), index=False)

                                df_exec_times = pd.DataFrame(execution_times, columns=["Lambda1", "Lambda2", "ExpectedReturnRate", "ExecutionTime"])
                                filename = f"{custom_prefix}execution_times_q_{q}_lambda1_{lambda1}_lambda2_{lambda2}_exp_rr_{expected_returns_rate}_freq_{self.freq}_tfreq_{t_freq}.csv"
                                df_exec_times.to_csv(os.path.join(results_dir, filename), index=False)
        
        print(q, lambda1, lambda2, expected_returns_rate, self.freq, t_freq)
        return df_portfolio_returns

    def save_files(self,df,arg, q,lambda1,lambda2,err, freq, t_freq ,filename, res_dir):
        # arg: string
        filename = arg + f"execution_times_q_{q}_lambda1_{lambda1}_lambda2_{lambda2}_exp_rr_{err}_freq_{freq}_tfreq_{t_freq}.csv"
        res_dir = res_dir
        df.to_csv(os.path.join(res_dir, filename), index=False)
        
        return df





    def RunBenchmarkBase(self, save_to_csv = False, print_debug = False):
        
        #Run Benchmark for base tickers of your choice.

        #Args: 
        #    1. save_to_csv: Boolean | Default = False | True if save file to desinated folder.
        #    2. print_debug: Boolean | Default = False | Print something for debugging purpose.

        #Return: 
        #    1. df_benchmarks_returns: dict | Return a database for each tickers performance


        data = self.data[self.bm_tickers]
        bms_adj = pd.Timestamp(self.bms)
        while bms_adj not in self.data[self.bm_tickers].index:
            bms_adj += BDay(1)  # Move to the next business day if necessary

        benchmark_values = {ticker: [self.B] for ticker in self.bm_tickers}  

        # Buy initial shares at the first available market date
        benchmark_shares = {
            ticker: [self.B / self.data[ticker].loc[data[ticker].index.asof(bms_adj)]]
            for ticker in self.bm_tickers
            } #TODO Remove asof

        # Loop through each rebalancing month
        for current_month in pd.date_range(self.bms, self.bme, freq = self.freq):
            next_month = pd.Timestamp(current_month) + BMonthBegin(1)
            #print(current_month)
            for ticker in self.bm_tickers:
                # Get last recorded value & shares
                last_value = benchmark_values[ticker][-1]
                last_shares = benchmark_shares[ticker][-1]

                # Adjust next_month to the closest available date in all_data[ticker] 
                if next_month not in self.data[ticker].index: # Offset holiday like labor day
                    next_month += BDay(1)   # Get the last valid business day

                if current_month != self.bms : 
                    benchmark_value = last_shares * self.data[ticker].loc[next_month] + self.new_invst
                    benchmark_values[ticker].append(benchmark_value)
                
                # Update new shares
                extra_shares = self.new_invst/self.data[ticker].loc[next_month]
                new_shares = last_shares + extra_shares
                #print(new_shares)
                benchmark_shares[ticker].append(new_shares) 

        # The first buy date is 01/01, the first return is on 02/01
        for ticker in self.bm_tickers:
            benchmark_values[ticker] = benchmark_values[ticker][:-1] # remove last element, only show first return in real price 

        # Convert benchmark values to DataFrame
        df_benchmarks = pd.DataFrame(benchmark_values, index=pd.date_range(self.bms, self.bme, freq=self.freq))

        # Calculate total return in %
        df_benchmarks_returns = ((df_benchmarks - self.new_invst - df_benchmarks.iloc[0])/ df_benchmarks.iloc[0]) * 100

        if save_to_csv:
            #TODO
            #TODO if no directory
            #TODO assigned a directory, if not create one and save it
            #TODO Define the subdirectory path
            if self.solver_type == 'classic':
                results_dir = "FAANG_Rebalance_results_Classic"
            elif self.solver_type == 'QAOA':
                results_dir = "FAANG_Rebalance_results_QAOA"
            elif self.solver_type == 'VQE':
                results_dir = "FAANG_Rebalance_results_VQE"

            # Ensure the directory exists
            os.makedirs(results_dir, exist_ok=True)

            # Save Database
            df_benchmarks_returns.to_csv(os.path.join(results_dir, "benchmarks_returns.csv"), index=False)

        if print_debug: 
            print("All_data['AAPL'].loc[data.index.asof(bms_adj)]")
            print(self.data["AAPL"].loc[data.index.asof(bms_adj)])
            print("Data.index.asof(bms_adj):")
            print(data.index.asof(bms_adj))
            print("Benchmark_shares: ", benchmark_shares)
        
        return df_benchmarks_returns
    
    def RunBenchmarkOptimizedOptPortfolio(self, save_to_csv = False):
        
        #https://pandas.pydata.org/docs/reference/api/pandas.tseries.offsets.DateOffset.html

        data = self.data[self.opt_tks]

        # Set seeds for reprodcibility
        seed = 123456

        # Get the last available stock prices (latest close prices)
        latest_prices = data.iloc[-1]  # Use last row to get most recent prices
        print("Latest Stock Prices:\n", latest_prices)
        prices = latest_prices.values

        # Estimate how many bits we need for binary encoding
        num_bits = Estimate_num_bits(self.B, prices)

        execution_times = [] 

        for lbda1 in self.lbda1: 
            for lbda2 in self.lbda2: 
                for _t_freq in self.t_freq:
                    for expected_returns_rate in self.exp_rr: # resonable period return rate
                        # Initialize list to store results for optimized portfolio
                        results_data = []
                        allocation_results = []
                        portfolio_value = []
                        budget = self.B

                        # Start timer
                        start_time = time.time()
                        
                        for current_month in pd.date_range(self.bms, self.bme, freq = self.freq): 
                            # Update rolling 3-year returns, r_i, covariance matrix
                            train_data = data.loc[current_month - pd.DateOffset(months = _t_freq): current_month] 
                            print("\n========== RunBenchmarkOptimizedOptPortfolio ==========")
                            print(f"TRAINING START MONTH: {current_month - pd.DateOffset(months = _t_freq)}")
                            print(f"CURRENT MONTH (rebalance date): {current_month}")
                            
                            returns, r_i, sigma = StatisticCalculatorRolling(train_data, freq = self.freq, print_out = False, plot = False)

                            # Get the closest past available trading day
                            closest_date = pd.Timestamp(current_month)
                            if closest_date not in self.data[self.bm_tickers].index:
                                closest_date += BDay(1)  # Move to the next business day if necessary
                            
                            # Fetch prices for the closest available market date
                            latest_prices = np.array(data.loc[closest_date].values) # Current month lastest price
                            
                            # Constructing our objective function
                            portfoilo_Hamiltonian = PortfolioOptimizationHamiltonian(
                                expected_returns = r_i,
                                covariances = sigma,
                                risk_factor = self.q,
                                budget = budget, 
                                lambda1 = lbda1, # Penalty coefficient 1
                                lambda2 = lbda2, # Penalty coefficient 2
                                prices = latest_prices, # Current month lastest price
                                max_binary_bits = num_bits,
                                expected_returns_rate = expected_returns_rate # expected return rate each time period from budget
                            )

                            #portfoilo_Hamiltonian
                            algorithm_globals.random_seed = 10598
                            qp = portfoilo_Hamiltonian.to_quadratic_program() # return a quadratic program
                            opt_results = self.solve_qp(qp, num_assets = len(r_i))

                            # Store results in a dictionary
                            results_data.append({
                                "Date": current_month.strftime("%Y-%m-%d"),
                                "Objective Value": opt_results.fval,
                                "Optimal Selection": list(opt_results.x)
                            })

                            latest_prices_zip = dict(zip(self.bm_tickers, latest_prices))
                            #before_optimized_current_value = sum(allocation_results) #TODO value of previous allocation to 

                            #print(f"before_optimized_current_value:{budget}")

                            share_allocation_classic = decode_results_rolling(result=opt_results, 
                                                                            tickers=self.opt_tks, 
                                                                            num_bits = num_bits, 
                                                                            current_month=current_month,
                                                                            latest_prices = latest_prices_zip)
                            
                            allocation_results.append(share_allocation_classic)
                            value_current_month = sum(share_allocation_classic[ticker] * latest_prices_zip[ticker] for ticker in data)
                            portfolio_value.append(value_current_month)


                            # === Calculate our portfolio value for the next month when we use allocation_results

                            # current 01/01 2024
                            # next 02/01 2024
                            # The next month is the first invest month, therefore, we will start put our money in it.
                            next_month = current_month + pd.DateOffset(months=1)
                            while next_month not in self.data[self.bm_tickers].index:
                                next_month += BDay(1)  # Move to the next business day if necessary
                            new_latest_prices = np.array(data.loc[next_month].values)

                            # Next period ticker price
                            new_latest_prices_zip = dict(zip(self.bm_tickers, new_latest_prices))

                            # Value on next month based on allocation from previous month
                            Allocation_cost = sum(share_allocation_classic[ticker] * new_latest_prices_zip[ticker] for ticker in data)

                            # Update our budget
                            budget = Allocation_cost + self.new_invst

                            #print(f"portfolio_value: {portfolio_value}\n")
                            
                        # End timer
                        end_time = time.time()  
                        exec_time = end_time - start_time  # Calculate duration
                        
                        # Store execution time for analysis
                        execution_times.append([lbda1, lbda2, expected_returns_rate, exec_time])

                        print(f"\n===== Final Portfolio Share Allocation =====")
                        print(pd.DataFrame(allocation_results).set_index('Date'))

                        print(f"Final value of the optimized portfoilo (cost of the last month): {(Allocation_cost)}")

                        total_return = ((portfolio_value[-1] - self.new_invst - portfolio_value[0]) / portfolio_value[0]) * 100
                        #print(f"Total return:{total_return}%")

                        # ===================== Post-Processing ===================== #
                        df_optimized_portfolio = pd.DataFrame(portfolio_value, index=pd.date_range(self.bms, self.bme, freq=self.freq))
                        df_portfolio_returns = ((df_optimized_portfolio - df_optimized_portfolio.iloc[0]) / df_optimized_portfolio.iloc[0]) * 100

                        # ===================== Save dataframe ===================== #
                        if save_to_csv:
                            if self.solver_type == 'classic':
                                results_dir = "FAANG_Rebalance_results_Classic"
                            elif self.solver_type == 'QAOA':
                                results_dir = "FAANG_Rebalance_results_QAOA"
                            elif self.solver_type == 'VQE':
                                results_dir = "FAANG_Rebalance_results_VQE"
                            # Save data
                            filename = f"Opt_Pf_{self.bms}_to_{self.bme}_train_{_t_freq}_exp_rr{expected_returns_rate}_lambda1_{lbda1}_lambda2_{lbda2}_{self.solver_type}.csv"
                            df_portfolio_returns.to_csv(os.path.join(results_dir, filename), index=False)

        if save_to_csv:
            if self.solver_type == 'classic':
                results_dir = "FAANG_Rebalance_results_Classic"
            elif self.solver_type == 'QAOA':
                results_dir = "FAANG_Rebalance_results_QAOA"
            elif self.solver_type == 'VQE':
                results_dir = "FAANG_Rebalance_results_VQE"
            # Save execution times as a CSV for tracking
            df_exec_times = pd.DataFrame(execution_times, columns=["Lambda1", "Lambda2", "ExpectedReturnRate", "ExecutionTime"])
            df_exec_times.to_csv(os.path.join(results_dir, "execution_times.csv"), index=False)

        return df_portfolio_returns
    
    #def solve_qp
    def solve_qp(self, qp, num_assets):
        
        #Solves a Quadratic Program (QP) using the specified solver type.

        #Parameters:
        #    qp (QuadraticProgram): The quadratic program to be solved.

        #Raises:
        #    ValueError: If an invalid solver type is provided.

        #Returns:
        #    results: The result of the optimization problem.
        
        if self.solver_type == 'classic':
            # Classic solver
            algorithm_globals.random_seed = 10598
            classical_algorithm = NumPyMinimumEigensolver()
            classical_eigensolver = MinimumEigenOptimizer(classical_algorithm)
            print(" --- Solving: Classic solver --- ")
            results = classical_eigensolver.solve(qp)

        elif self.solver_type == 'QAOA':
            # QAOA solver
            algorithm_globals.random_seed = 10598
            qaoa_optimizer = COBYLA(maxiter=250)
            qaoa_sampler = QAOA(sampler=Sampler(), optimizer=qaoa_optimizer, reps=3)
            qaoa_res = MinimumEigenOptimizer(qaoa_sampler)
            print(" --- Solving: Quantum solver (QAOA) --- ")
            results = qaoa_res.solve(qp)

        elif self.solver_type == 'VQE':
            # set VQE optimizer
            algorithm_globals.random_seed = 10598
            vqe_optimzer = COBYLA(maxiter=500)
            vqe_ansatz = TwoLocal(num_assets, "ry", "cz", reps = 3, entanglement="full")
            vqe_sampler = SamplingVQE(sampler=Sampler(), ansatz=vqe_ansatz, optimizer=vqe_optimzer)
            vqe_res = MinimumEigenOptimizer(vqe_sampler)
            print(" --- Solving: Quantum solver (VQE) --- ")
            results = vqe_res.solve(qp)

        else:
            raise ValueError(f"Invalid solver type '{self.solver_type}'. Supported types: 'classic', 'QAOA', 'VQE'.")

        return results