import os
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from DataManager import DataManager
from PortfolioManager import PortfolioManager 

history_tickers = ['AAPL','NVDA','MSFT','AMZN','VOO','XLF','XLE','XLV'] 
history_start_date = '2018-01-01'
history_end_date = '2025-03-01'

DM = DataManager(tickers = history_tickers,
                 start_date = history_start_date,
                 end_date = history_end_date)

DM.fetch_yahoo_finance_data(price = 'Close',
                            fetch=False,
                            save_to_file=False)

data = DM.load_data(path=
                    "all_data_2025-03-19_02-01-51AAPL_NVDA_MSFT_AMZN_VOO_XLF_XLE_XLV2018-01-01to2025-03-01.csv")

#assets_portfolio = ['AAPL','MSFT','AMZN','XLF','XLE']
assets_portfolio = ['AAPL','MSFT','AMZN']
assets_benchmark = ['VOO']

PM = PortfolioManager(data = data,
                      budget = 10000, 
                      new_invest = 0, 
                      assets_portfolio = assets_portfolio,
                      assets_benchmark = assets_benchmark,)

bm_start_date = "2025-01-01"
bm_end_date = "2025-02-05"
save_path = "VOO_encoding_and_lambda_study"


bm_base = PM.run_benchmark_simulator(start = bm_start_date,
                                     end = bm_end_date,
                                     save = True,
                                     save_path = save_path,
                                     freq = '21B')

# Study 1, encoding vs lambda: encoding verses the budget

k_list = [2, 3, 6]
lambda1s = [1E3, 1E6, 1E8]
#lambda1s = [1E-3, 1E-2, 1E-1, 1]
#qs = [1, 0.1, 0.01, 0.001]
qs = [0]
solvers = ['classic','QAOA','SamplerVQE']
c = 0

for q in qs:
    for k in k_list:
        for lambda1 in lambda1s:
            c = 0
            while c < 60:
                print(c)
                for solver in solvers:
                    try:
                        opt_res = PM.run_portfolio_optimization(start = bm_start_date,
                                                            end = bm_end_date,
                                                            save = True,
                                                            save_path = save_path,
                                                            custom_prefix = f"{c}_",
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
                c += 1

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
#plt.show()