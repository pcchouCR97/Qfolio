import sys
import os
import time
import numpy as np
import pandas as pd
import concurrent.futures
from pandas.tseries.offsets import BDay
from pandas.tseries.frequencies import to_offset
from print_n_plots import decode_results_rolling_k
from utilities import estimate_num_bits_basek, StatisticCalculatorRollingD
from PortfolioOptimization_v2 import PortfolioOptimizationHamiltonian
from pandas.tseries.frequencies import to_offset
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_algorithms.utils import algorithm_globals
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms import QAOA, SamplingVQE, VQE
from qiskit_aer.primitives import Sampler, Estimator
from qiskit.circuit.library import TwoLocal

class PortfolioOptimizer():
    def __init__(self, 
                 assets, 
                 data, 
                 budget, 
                 new_investment,
                 k,
                 lambda1,
                 q,
                 freq,
                 t_freq,
                 H_scale,
                 solver_type):
        self.assets = assets
        self.data = data
        self.B = budget
        self.new_invst = new_investment
        self.k = k
        self.lambda1 = lambda1
        self.q = q
        self.freq = freq
        self.t_freq = t_freq
        self.H_scale = H_scale
        self.solver_type = solver_type
        
    def optimize(self,
                 start_date,
                 end_date,
                 save, 
                 save_path,
                 custom_prefix): # training frequency
        
        data = self.data[self.assets]
        
        # Get the last available stock prices (latest close prices)
        latest_prices = data.iloc[-1]  # Use last row to get most recent prices
        
        prices = latest_prices.values

        # Estimate how many bits we need for binary encoding
        num_bits = estimate_num_bits_basek(self.B, prices, k = self.k)
        print(f"budget: {self.B}, price: {prices}, k: {self.k}")
        print(num_bits)

        execution_times = []
        # Step through every nth date using freq logic
        # Rebalance period
        offset = to_offset(self.freq)
        
        step = offset.n if hasattr(offset, 'n') else 1

        bms = pd.Timestamp(start_date)
        bme = pd.Timestamp(end_date)
        bms_adj = bms
        while bms_adj not in data.index:
            bms_adj += BDay(1)

        # Get the list of all valid dates
        valid_dates = self.data.loc[bms:bme].index

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
        # Train frequency
        train_offset = to_offset(self.t_freq)
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
                                                            freq = self.t_freq, 
                                                            print_out = False, 
                                                            plot = False)
            
            # Get the closest past available trading day
            closest_date = pd.Timestamp(crp)
            if closest_date not in data.index:
                closest_date += BDay(1)  # Move to the next business day if necessary
            
            # Fetch prices for the closest available market date
            latest_prices = np.array(data.loc[closest_date].values) # Current month lastest price
            
            # Constructing our objective function
            portfoilo_Hamiltonian = PortfolioOptimizationHamiltonian(
                expected_returns = r_i,
                covariances = sigma,
                budget = budget,
                q = self.q,
                lambda1 = self.lambda1, # Penalty coefficient 1
                prices = latest_prices, # Current month lastest price
                max_binary_bits = num_bits,
            )
            
            #portfoilo_Hamiltonian
            qp = portfoilo_Hamiltonian.to_quadratic_program(k = self.k,
                                                              H_scale = self.H_scale) # return a quadratic program
            # Debug
            #print(qp)

            #sys.exit("Stopping execution here")
            opt_results = self.solve_qp(qp, num_assets = len(r_i))
            
            if opt_results is None:
                print(f"Optimization timed out for {crp}. Skipping this parameter set.")
                raise RuntimeError(f"Optimization failed at {crp}. Skipping this run.")

            # Store results in a dictionary
            results_data.append({
                "Date": crp.strftime("%Y-%m-%d"),
                "Objective Value": opt_results.fval,
                "Optimal Selection": list(opt_results.x)
            })

            latest_prices_zip = dict(zip(self.assets, latest_prices))

            share_allocation_classic = decode_results_rolling_k(result=opt_results,
                                                                assets=self.assets, 
                                                                num_bits = num_bits, 
                                                                crp=crp,
                                                                latest_prices = latest_prices_zip,
                                                                k = self.k)
            
            allocation_results.append(share_allocation_classic)
            val_crp = sum(share_allocation_classic[ticker] * latest_prices_zip[ticker] for ticker in data)
            portfolio_value.append(val_crp) # portfolio value after applying allocation_results
            cost_diff.append(val_crp - budget) # let's see how strong the budget constraint leads us!
        
            # MWR (next period)
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
                while nrp not in data.index:
                    nrp += BDay(1)  # Move to the next business day if necessary
                new_latest_prices = np.array(data.loc[nrp].values)

            # Next period ticker price
            new_latest_prices_zip = dict(zip(self.assets, new_latest_prices))

            # Value on next month based on allocation from previous month at the sametime
            val_nrp = sum(share_allocation_classic[ticker] * new_latest_prices_zip[ticker] for ticker in data)

            # Update our budget
            budget = val_nrp + self.new_invst
            
            print(f"==== this {crp} train loop end ====")
            
        # End timer
        end_time = time.time()  
        exec_time = end_time - start_time  # Calculate duration
        
        # Store execution time for analysis
        execution_times.append([self.lambda1, exec_time])

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
        if save:
            # set result directory
            results_dir = save_path   
            # Save data
            filename = f"{custom_prefix}port_return_B_{self.B}_new_invest{self.new_invst}_k_{self.k}_q_{self.q}_lambda1_{self.lambda1}_freq_{self.freq}_tfreq_{self.t_freq}_H_scale_{self.H_scale}_solver_type{self.solver_type}.csv"
            
            df_portfolio_returns.to_csv(os.path.join(results_dir, filename), index=False)

            # Save difference data 
            df_cost_diff = pd.DataFrame(cost_diff, index=crps)
            filename = f"{custom_prefix}cost_diff_B_{self.B}_new_invest{self.new_invst}_k_{self.k}_q_{self.q}_lambda1_{self.lambda1}_freq_{self.freq}_tfreq_{self.t_freq}_H_scale_{self.H_scale}_solver_type{self.solver_type}.csv"
            
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
            filename = f"{custom_prefix}allocation_B_{self.B}_new_invest{self.new_invst}_k_{self.k}_q_{self.q}_lambda1_{self.lambda1}_freq_{self.freq}_tfreq_{self.t_freq}_H_scale_{self.H_scale}_solver_type{self.solver_type}.csv"
            
            allocation_results.to_csv(os.path.join(results_dir, filename), index=False)

            df_exec_times = pd.DataFrame(execution_times, columns=["Lambda1", "ExecutionTime"])
            filename = f"{custom_prefix}execution_times_B_{self.B}_new_invest{self.new_invst}_k_{self.k}_q_{self.q}_lambda1_{self.lambda1}_freq_{self.freq}_tfreq_{self.t_freq}_H_scale_{self.H_scale}_solver_type{self.solver_type}.csv"

            df_exec_times.to_csv(os.path.join(results_dir, filename), index=False)
        
        return df_portfolio_returns
    
    def solve_qp(self, qp, num_assets):
        
        #Solves a Quadratic Program (QP) using the specified solver type.

        #Parameters:
        #    qp (QuadraticProgram): The quadratic program to be solved.

        #Raises:
        #    ValueError: If an invalid solver type is provided.

        #Returns:
        #    results: The result of the optimization problem.
        
        timeout_seconds = 60  # 1 minutes

        def run_solver():
            algorithm_globals.random_seed = 10598

            if self.solver_type == 'classic':
                classical_algorithm = NumPyMinimumEigensolver()
                classical_eigensolver = MinimumEigenOptimizer(classical_algorithm)
                print(" --- Solving: Classic solver --- ")
                return classical_eigensolver.solve(qp)

            elif self.solver_type == 'QAOA':
                qaoa_optimizer = COBYLA(maxiter=250)
                qaoa_sampler = QAOA(sampler=Sampler(), optimizer=qaoa_optimizer, reps=3)
                qaoa_res = MinimumEigenOptimizer(qaoa_sampler)
                print(" --- Solving: Quantum solver (QAOA) --- ")
                return qaoa_res.solve(qp)

            elif self.solver_type == 'SamplerVQE':
                vqe_optimzer = COBYLA(maxiter=500)
                vqe_ansatz = TwoLocal(num_assets, "ry", "cz", reps=3, entanglement="full")
                vqe_sampler = SamplingVQE(sampler=Sampler(), ansatz=vqe_ansatz, optimizer=vqe_optimzer)
                vqe_res = MinimumEigenOptimizer(vqe_sampler)
                print(" --- Solving: Quantum solver (SamplerVQE) --- ")
                return vqe_res.solve(qp)
            
            else:
                raise ValueError(f"Invalid solver type '{self.solver_type}'. Supported types: 'classic', 'QAOA', 'SamplerVQE'.")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_solver)
            try:
                results = future.result(timeout=timeout_seconds)
            except concurrent.futures.TimeoutError:
                print(f"Solver timed out after {timeout_seconds // 60} minutes.")
                results = None

        return results