import os
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from DataManager import DataManager
from PortfolioManager import PortfolioManager 
import sys

history_tickers = ['PLTR','ORCL','HOOD','VOO','NVDA'] 
history_start_date = '2024-01-01'
history_end_date = '2025-07-18'

DM = DataManager(tickers = history_tickers,
                 start_date = history_start_date,
                 end_date = history_end_date)

DM.fetch_yahoo_finance_data(price = 'Close', # adjusted close price
                            fetch=False,
                            save_to_file=False)

#sys.exit("Data fetching complete. Please check the saved files before proceeding.")

# bull/ bear
data = DM.load_data(path="all_data_2025-07-18_01-12-43_PLTR_ORCL_HOOD_VOO_NVDA_2024-01-01to2025-07-18.csv")

# 3yd
#data = DM.load_data(path="all_data_2025-07-15_20-48-23_AAPL_MSFT_AMZN_VOO_2021-10-01to2025-06-06.csv")
assets_portfolio = ['PLTR','ORCL','HOOD','NVDA']
assets_benchmark = ['VOO']

PM = PortfolioManager(data = data,
                      budget = 10000, 
                      new_invest = 1000, 
                      assets_portfolio = assets_portfolio,
                      assets_benchmark = assets_benchmark,)

# 1 yd
bm_start_date = "2024-03-12"
bm_end_date = "2025-07-18"

freq = '21B'  # 21 business days for monthly rebalancing

bm_base = PM.run_benchmark_simulator(start = bm_start_date,
                                     end = bm_end_date,
                                     save = True,
                                     save_path = "VOO_testbed_data",
                                     freq = '21B')

opt_res = PM.run_portfolio_optimization(start = bm_start_date,
                                     end = bm_end_date,
                                     save = True,
                                     save_path = "VOO_testbed_data",
                                     custom_prefix = f"{0}_",
                                     k = 2, # encoding method
                                     lambda1 = 1E3, # budget penalty
                                     q = 1E-3, # risk aversion coefficient
                                     freq = '21B', # rebalance frequency
                                     t_freq = '21B',  # training frequency
                                     H_scale = 1, # objective function scaling
                                     solver_type = 'classic') # sovler type, suppoer QAOA or classic

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