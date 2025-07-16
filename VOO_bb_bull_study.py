import os
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from DataManager import DataManager
from PortfolioManager import PortfolioManager 

history_tickers = ['AAPL','MSFT','AMZN','VOO'] 
history_start_date = '2021-10-01'
history_end_date = '2024-12-31'

DM = DataManager(tickers = history_tickers,
                 start_date = history_start_date,
                 end_date = history_end_date)

DM.fetch_yahoo_finance_data(price = 'Close',
                            fetch=False,
                            save_to_file=False)

data = DM.load_data(path="all_data_2025-07-15_16-11-49_AAPL_MSFT_AMZN_VOO_2021-10-01to2024-12-31.csv")
assets_portfolio = ['AAPL','MSFT','AMZN']
assets_benchmark = ['VOO']

PM = PortfolioManager(data = data,
                      budget = 10000, 
                      new_invest = 1000, 
                      assets_portfolio = assets_portfolio,
                      assets_benchmark = assets_benchmark,)

# Bull
bm_start_date = "2023-10-27"
bm_end_date = "2024-10-24"

save_path = "VOO_bb_bull_study"

bm_base = PM.run_benchmark_simulator(start = bm_start_date,
                                     end = bm_end_date,
                                     save = True,
                                     save_path = save_path,
                                     freq = '21B')
k_list = [2]
lambda1s = [1E3]
qs = [0.001, 0.1, 1, 5]
solvers = ['classic','QAOA']#,'SamplerVQE']

for solver in solvers:
    for q in qs:
        for k in k_list:
            for lambda1 in lambda1s:
                try: # using PortfolioOptimization_v2
                    opt_res = PM.run_portfolio_optimization(start = bm_start_date,
                                                        end = bm_end_date,
                                                        save = True,
                                                        save_path = save_path,
                                                        custom_prefix = "",
                                                        k = k, # encoding method
                                                        lambda1 = lambda1, # budget penalty
                                                        q = q, # risk aversion coefficient
                                                        freq = '21B', # rebalance frequency
                                                        t_freq = '21B',  # training frequency
                                                        H_scale = 1, # objective function scaling
                                                        solver_type = solver) # sovler type
                except RuntimeError as e:
                    print(f"[SKIPPED] {solver}, q={q}, k={k}, λ={lambda1} : {e}")
                    continue
# Study 1, encoding vs lambda: encoding verses the budget
# Study 2 run (21B,21B) with (63B,63B) 4 combinations with qs and appropreate lambdas, good k.
#

# disable plotting
"""
# ===================== Plotting ===================== #
# Plot optimized portfolio vs benchmarks
plt.figure(figsize=(12, 6))
plt.plot(opt_res, label="Optimized Portfolio", linewidth=2, color='black')

bm_tickers = assets_portfolio + assets_benchmark

for ticker in bm_tickers:
    if ticker in assets_benchmark:
        plt.plot(bm_base[ticker], label=ticker, linewidth=2, linestyle = "-")
    else:
        plt.plot(bm_base[ticker], label=ticker, linestyle="--")

plt.title("Optimized Portfolio vs. Benchmarks " + ", ".join(bm_tickers))
plt.xlabel("Time")
plt.ylabel("Return (%)")

# Format x-axis to show full dates (YYYY-MM-DD)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # Show one label per month

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

plt.legend()
plt.grid(True)
plt.show()
"""