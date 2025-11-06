import sys
import os
import time
import numpy as np
import pandas as pd
import concurrent.futures
from pandas.tseries.offsets import BDay
from qfolio.utils.print_n_plots import decode_results_AMSP
from qfolio.utils.utilities import estimate_num_bits_basek, StatisticCalculatorRollingD_AMSP, estimate_num_bits_basek, AMSP_estimator
from qfolio.optimization.hamiltonian import PortfolioOptimizationHamiltonian
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_algorithms.utils import algorithm_globals
from qiskit_optimization.algorithms import MinimumEigenOptimizer, WarmStartQAOAOptimizer, RecursiveMinimumEigenOptimizer, CplexOptimizer, GurobiOptimizer  
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit_algorithms import QAOA, SamplingVQE
from qiskit_aer.primitives import Sampler
from qiskit.circuit.library import TwoLocal, efficient_su2, EfficientSU2


class PortfolioOptimization_v2_3_AMSP():
    def __init__(self, 
                 data, 
                 k,
                 lambda1,
                 q,
                 H_scale,
                 solver_type):
        self.data = data
        self.k = k
        self.lambda1 = lambda1
        self.q = q
        self.H_scale = H_scale
        self.solver_type = solver_type
        
    def optimize_single_period(self,
                               assets,
                               budget,
                               current_date,
                               train_start,
                               train_end,
                               latest_prices_dict):
        
        data = self.data[assets]
        
        # Use the passed dictionary of current open prices for allocation
        # The order of prices must match the order of assets/columns in `data`
        prices = [latest_prices_dict[asset] for asset in assets]

        if self.k == 2: num_bits = 7
        else: num_bits = estimate_num_bits_basek(budget, prices, k = self.k)

        train_data = data.loc[train_start:train_end]

        # --- DEBUG PRINT ---
        print(f"[DEBUG] Shape of train_data for optimization: {train_data.shape}")
        if train_data.empty:
            print("[DEBUG] train_data is EMPTY. Slicing failed.")
        # --- END DEBUG ---

        
        lots, AMSP = AMSP_estimator(np.array(prices), 
                                    max_binary_bits = num_bits, 
                                    budget = budget,
                                    k = 2)
        
        returns, r_i, sigma = StatisticCalculatorRollingD_AMSP(train_data, 
                                                        train_start=train_data.index[0],
                                                        train_end=train_data.index[-1],
                                                        lots = lots,
                                                        print_out = True, 
                                                        plot = False)

        portfoilo_Hamiltonian = PortfolioOptimizationHamiltonian(
            expected_returns = r_i,
            covariances = sigma,
            budget = budget,
            q = self.q,
            lambda1 = self.lambda1,
            prices = prices, # Use the open prices here
            max_binary_bits = num_bits,
        )
        qp = portfoilo_Hamiltonian.to_quadratic_program(k = self.k,H_scale = self.H_scale) 
        #print(qp.export_as_lp_string())
        #print(f"Number of variables in QP: {qp.get_num_vars()}")
        #sys.exit("test exit")
        opt_results = self.solve_qp(qp, num_assets = len(assets))
        
        if opt_results is None or opt_results.status.name != 'SUCCESS':
            tr = 0
            print(f"Optimization failed or timed out for {current_date}, retry {tr+1}." )
            for tr in range(5):
                time.sleep(5) # wait before retry
                opt_results = self.solve_qp(qp, num_assets = len(assets))
                if opt_results is not None and opt_results.status.name == 'SUCCESS':
                    print(f"Retry {tr+1} succeeded.")
                    break
                else:
                    print(f"Retry {tr+1} failed.")
                    tr += 1
                
            print(f"Optimization failed or timed out for {current_date} after {tr+1} tries. Skipping this period.")
            return {
                "allocation": {asset: 0 for asset in assets},
                "value": 0,
                "date": current_date
            }

        latest_prices_zip = dict(zip(assets, prices))

        share_allocation = decode_results_AMSP(result=opt_results,
                                                            assets=assets, 
                                                            num_bits = num_bits, 
                                                            crp=current_date,
                                                            latest_prices = latest_prices_zip,
                                                            k = 2,
                                                            lots= lots,
                                                            AMSP = AMSP)
        # The allocation dictionary might include a 'Date' key, which is not a ticker.
        # We remove it before returning the allocation.
        if 'Date' in share_allocation:
            del share_allocation['Date']
        
        portfolio_value = sum(share_allocation[ticker] * latest_prices_zip[ticker] for ticker in assets)
        
        return {
            "allocation": share_allocation,
            "value": portfolio_value,
            "date": current_date
        }

    def solve_qp(self, qp, num_assets):
        # Set timeout based on solver type
        # Quantum solvers need much more time for circuit transpilation + optimization
        if self.solver_type in ['QAOA', 'QAOA_shots', 'QAOA_MPS', 'QAOA_warm_start_recursive',
                                'SamplerVQE', 'QVE_MPS']:
            timeout_seconds = 1800  # 30 minutes for quantum (28 qubits needs this)
        elif self.solver_type in ['SCIP', 'exact_miqp']:
            timeout_seconds = 300  # 5 minutes for classical exact solvers
        else:
            timeout_seconds = 60  # 1 minute for other solvers

        def run_solver():
            algorithm_globals.random_seed = 10598

            if self.solver_type == 'classic':
                # Classical exact solver using NumPyMinimumEigensolver
                classical_algorithm = NumPyMinimumEigensolver()
                classical_eigensolver = MinimumEigenOptimizer(classical_algorithm)
                return classical_eigensolver.solve(qp)
                       
            elif self.solver_type == 'QAOA':
                # Default QAOA with statevector simulation
                print(f" --- Starting QAOA for {qp.get_num_vars()} qubits --- ")
                
                qaoa_optimizer = COBYLA(maxiter=100)
                qaoa_sampler = QAOA(sampler=Sampler(), optimizer=qaoa_optimizer, reps=1)
                qaoa_res = MinimumEigenOptimizer(qaoa_sampler)
                return qaoa_res.solve(qp)
            
            elif self.solver_type == 'QAOA_shots':
                # Pure shot-based sampling with AerSimulator
                # Uses 512 shots per evaluation to balance accuracy and speed

                print(f" --- Starting QAOA_shots for {qp.get_num_vars()} qubits --- ")

                backend_opts = {
                    "method": "automatic"  # Let AerSimulator choose optimal sampling method
                }

                run_opts = {
                    "shots": 512  # Number of measurement samples
                }

                # Transpiler options for LOCAL AerSimulator (speeds up 5-10×)
                transpile_opts = {
                    "optimization_level": 0,  # 0=fastest (30-120 sec), 1=fast, 2=default, 3=slowest
                    "seed_transpiler": 10598
                }

                qaoa_optimizer = COBYLA(maxiter=30)  # Reduced from 100 for faster convergence
                qaoa_sampler = QAOA(
                    sampler=Sampler(backend_options=backend_opts,run_options=run_opts,
                        transpile_options=transpile_opts # Speeds up transpilation
                    ),
                    optimizer=qaoa_optimizer,
                    reps=1
                )

                qaoa_res = MinimumEigenOptimizer(qaoa_sampler)
                print(" --- Starting QAOA optimization (30 iterations) --- ")
                return qaoa_res.solve(qp)

            elif self.solver_type == 'SamplerVQE':
                # QUBO creates a diagonal Hamiltonian (only Z operators)
                # Therefore, dont use estimators that assume general Hamiltonians with X,Y terms.
                # Sampler is optimal for diagonal Hamiltonians (evaluates energy directly from bitstrings)
                # Estimator is for non-diagonal Hamiltonians (needs to compute ⟨ψ|H|ψ⟩ with X/Y operators)
                # Using Estimator would be slower and more complex with no benefit
                # qiskit: "SamplingVQE/QAOA uses the sampler primitive because it's optimized for diagonal Hamiltonians, where energy can be evaluated directly from sampled bitstrings." Conclusion: Stick with Sampler, just configure it properly to avoid statevector.
                vqe_optimzer = COBYLA(maxiter=100)
                #TODO: maxiter tuning, speed 
                #TODO: ansatz tuning, Twolocal vs EfficientSU2
                vqe_ansatz = TwoLocal(num_assets, "ry", "cz", reps = 1, entanglement="full")
                #vqe_ansatz = EfficientSU2(num_assets, reps=1)
                vqe_sampler = SamplingVQE(sampler=Sampler(), ansatz=vqe_ansatz, optimizer=vqe_optimzer)
                vqe_res = MinimumEigenOptimizer(vqe_sampler)
                print(" --- Solving: Quantum solver (SamplerVQE) --- ")
                return vqe_res.solve(qp)
            
            elif self.solver_type == 'QAOA_MPS_optimized':
                backend_opts = {
                    "method": "matrix_product_state",
                    "matrix_product_state_max_bond_dimension": 64,  # Controls memory
                    "matrix_product_state_truncation_threshold": 1e-5,  # Truncate small singular values
                    "max_parallel_threads": 1  # Reduce parallelism to save memory
                }
        
                qaoa_optimizer = COBYLA(maxiter=100)
                
                # Use AerSimulator directly with MPS
                from qiskit_aer import AerSimulator
                backend = AerSimulator(**backend_opts)
                
                qaoa_sampler = QAOA(
                    sampler=Sampler(backend=backend),
                    optimizer=qaoa_optimizer,
                    reps=1  # Keep reps=1 for low entanglement
                )
                qaoa_res = MinimumEigenOptimizer(qaoa_sampler)
                print(" --- Solving: Quantum solver (QAOA_MPS_optimized) --- ")
                return qaoa_res.solve(qp)
            
            elif self.solver_type == 'VQE_MPS': 
                backend_opts = {"method": "matrix_product_state"}
                vqe_optimzer = COBYLA(maxiter=100)
                vqe_ansatz = TwoLocal(num_assets, "ry", "cz", reps = 1, entanglement="full")
                vqe_sampler = SamplingVQE(sampler=Sampler(backend_options=backend_opts), ansatz=vqe_ansatz, optimizer=vqe_optimzer)
                vqe_res = MinimumEigenOptimizer(vqe_sampler)
                print(" --- Solving: Quantum solver (VQE_MPS) --- ")
                return vqe_res.solve(qp)


            elif self.solver_type == 'SCIP' or self.solver_type == 'exact_miqp':
                from qfolio.optimization.ClassicalExactSolver import ExactMIQPSolver
                scip_solver = ExactMIQPSolver()
                print(" --- Solving: Classical MIQP solver (CVXPY+SCIP) --- ")
                return scip_solver.solve(qp)

            else:
                raise ValueError(f"Invalid solver type '{self.solver_type}'. Supported types: 'classic', 'QAOA', 'SamplerVQE', 'SCIP', 'exact_miqp', 'QAOA_MPS'.")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_solver)
            try:
                results = future.result(timeout=timeout_seconds)
            except concurrent.futures.TimeoutError:
                print(f"Solver timed out after {timeout_seconds // 60} minutes.")
                results = None

        return results
