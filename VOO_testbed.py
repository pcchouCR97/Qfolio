import os
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from DataManager import DataManager
from PortfolioManager import PortfolioManager 

history_tickers = ['AAPL','MSFT','AMZN','VOO'] 
history_start_date = '2022-02-01'
history_end_date = '2025-06-06'

DM = DataManager(tickers = history_tickers,
                 start_date = history_start_date,
                 end_date = history_end_date)

DM.fetch_yahoo_finance_data(price = 'Close',
                            fetch=True,
                            save_to_file=True)
# bull/ bear
data = DM.load_data(path="all_data_2025-07-15_16-11-49_AAPL_MSFT_AMZN_VOO_2021-10-01to2024-12-31.csv")

# 3yd
#data = DM.load_data(path="all_data_2025-07-15_20-48-23_AAPL_MSFT_AMZN_VOO_2021-10-01to2025-06-06.csv")
assets_portfolio = ['AAPL','MSFT','AMZN']
assets_benchmark = ['VOO']

PM = PortfolioManager(data = data,
                      budget = 10000, 
                      new_invest = 1000, 
                      assets_portfolio = assets_portfolio,
                      assets_benchmark = assets_benchmark,)

# 3yd 
bm_start_date = "2022-06-06"
bm_end_date = "2025-06-06"

# Bull
#bm_start_date = "2023-10-27"
#bm_end_date = "2024-10-24"

# Bear
#bm_start_date = "2022-01-04"
#bm_end_date = "2022-10-13"

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
                                     solver_type = 'classic') # sovler type

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