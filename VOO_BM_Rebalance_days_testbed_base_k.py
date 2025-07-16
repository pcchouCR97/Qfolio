import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import time
from pandas.tseries.offsets import BDay, BMonthBegin
from pandas.tseries.frequencies import to_offset
from print_n_plots import decode_results_rolling_k
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms.utils import algorithm_globals
from PortfolioOptimization import PortfolioOptimizationHamiltonian
from utilities import estimate_num_bits_basek, StatisticCalculatorRollingD
from StockDataCollector import StockDataCollector

# ===================== General Settings ===================== #
# Define tickers you want to analysis with its history date
# AAPL,AMZN,META,NFLX,QQQ
# VOO is the SP500, select 'AAPL','NVDA','MSFT','AMZN','VOO' for comparison
# VOO, XLF
all_tickers = ['AAPL','NVDA','MSFT','AMZN','VOO','XLF','XLE','XLV'] 
history_start_date = '2018-01-01'
history_end_date = '2025-03-01'

fetch_all_data = StockDataCollector(tickers=all_tickers,
                                    start_date=history_start_date,
                                    end_date=history_end_date,
                                    price='Close',
                                    save_to_database = False,
                                    fetch = False).FetchYahooFinanceData()


# read data 
all_data = pd.read_csv('all_data_2025-03-19_02-01-51AAPL_NVDA_MSFT_AMZN_VOO_XLF_XLE_XLV2018-01-01to2025-03-01.csv', index_col=0, parse_dates=True)
#all_data = pd.read_csv('all_data_2025-03-30_10-58-36GE_HWM_AAPL_RTX_VOO2018-01-01to2025-03-15.csv', index_col=0, parse_dates=True)

# Define our data
init_budget = 10000 # set budget
new_investment = 300 # Define new investment amount every month


# Set train tickers you to estalish optimized portfolio
#tickers = ['RTX','GE','HWM']
#tickers = ['BRK-B','JPM','V']
tickers = ['AAPL','MSFT','AMZN']#,'XLF']

# Define Train data column
data = all_data[tickers] 

# Date frequency
freq = '21B' # Start of the business days/ month, how frequet you want to rebalance
train_freq = '21B' # This is the rolling training time span # 60 is giving a crazy number lol
# You can rebalance each 'freq' period using trained date for each 'train_freq' data.

# ===================== Benchmark Settings ===================== #

# Set stock tickers you want to benchmark
bm_bud = init_budget # (b)ench(m)ark (b)udget
#target_tickers = ['XLF','XLV','XLE'] 
target_tickers = ['VOO']
bm_tickers = tickers + target_tickers

# Train and benchmark setting
# Time span settings
bms = "2024-03-01" # (b)ench(m)ark (s)tart date
bme = "2025-03-19" # (b)ench(m)ark (e)nd date

# Ensure bms is adjusted to the first available market date
bms_adj = pd.Timestamp(bms) # (b)ench(m)ark (s)tart date (adj)ust.
while bms_adj not in all_data[bm_tickers].index:
    bms_adj += BDay(1)  # Move to the next business day if necessary
bm_val = {ticker: [bm_bud] for ticker in bm_tickers}  
print(bms_adj)

# Buy initial shares at the first available market date
bm_shares = {
    ticker: [bm_bud / all_data[ticker].loc[bms_adj]]
    for ticker in bm_tickers
    }

# Step through every nth date using freq logic
offset = to_offset(freq)
step = offset.n if hasattr(offset, 'n') else 1

# Get the list of all valid dates
valid_dates = all_data.loc[bms:bme].index

print(valid_dates)
# Get crps (current rebalancing points) and nrps (next periods)
crps = valid_dates[::step] # (c)urrent (r)eturn (p)eriod: current rebalance date
nrps = valid_dates[step::step] # (n)ext (r)eturn (p)eriod: next rebalance date 
print(step)
print(crps)
#sys.exit("Stopping execution here")

for crp, nrp in zip(crps, nrps):
    #print(f"current {crp}, next {nrp}")
    
    for ticker in bm_tickers:
        # Get last recorded value & shares
        last_value = bm_val[ticker][-1]
        last_shares = bm_shares[ticker][-1] # get 01/01 shares
        
        if crp != bms : 
            benchmark_value = last_shares * all_data[ticker].loc[nrp] + new_investment
            bm_val[ticker].append(benchmark_value)
        
        # Update new shares
        extra_shares = new_investment/all_data[ticker].loc[nrp]
        new_shares = last_shares + extra_shares
        #print(new_shares)
        bm_shares[ticker].append(new_shares) 

for ticker in bm_tickers:
    bm_val[ticker] = bm_val[ticker] # remove last element, only show first return in real price 

# Convert benchmark values to DataFrame
df_benchmarks = pd.DataFrame(bm_val, index=crps, columns=[ticker for ticker in bm_tickers])

invested_amounts = [init_budget + i * new_investment for i in range(len(df_benchmarks))]

# Convert to Series aligned with index
invested_series = pd.Series(invested_amounts, index=df_benchmarks.index)

# Step 2: Broadcast to match all columns (tickers)
invested_df = pd.DataFrame({col: invested_series for col in df_benchmarks.columns})

# Cumulative return accounting for new capital
df_benchmarks_returns = ((df_benchmarks - invested_df) / invested_df) * 100

# Define the subdirectory path
results_dir = "FAANG_Rebalance_results_days_test"

# Ensure the directory exists
os.makedirs(results_dir, exist_ok=True)

# Save Database
df_benchmarks_returns.to_csv(os.path.join(results_dir, "benchmarks_returns.csv"), index=False)

# ===================== Optimized Portfolio Settings ===================== #
# Set seeds for reprodcibility
seed = 123456

# Get the last available stock prices (latest close prices)
latest_prices = data.iloc[-1]  # Use last row to get most recent prices
#print("Latest Stock Prices:\n", latest_prices)
prices = latest_prices.values

init_budget = init_budget # scaling doesnt matter
latest_prices = latest_prices # scaling doesnt matter

# Estimate how many bits we need for binary encoding
k = 4
num_bits = estimate_num_bits_basek(init_budget, prices, k = k)
print(f"budget: {init_budget}, price: {prices}, k: {k}")
print(num_bits)
#sys.exit("Stopping execution here")

execution_times = []
q = 0.001

for lambda1 in [1E2]: # budget
    for lambda2 in [1E4]: # return
        for expected_returns_rate in [0]: # resonable period return rate
            # Initialize list to store results for optimized portfolio
            results_data = []
            allocation_results = []
            portfolio_value = []
            return_ = []
            budget = init_budget
            cost_diff = []
            print(budget)

            # Makesure train_freq is not freq
            train_offset = to_offset(train_freq)
            train_step = train_offset.n if hasattr(train_offset, 'n') else 1
        
            # Start timer
            start_time = time.time()
            for i in range(len(crps)):
                crp = crps[i]
                nrp = nrps[i] if i < len(nrps) else None  # handle last crp without nrp

                train_start = crp - train_offset
                train_end = crp

                idx = data.index.get_loc(crp)
                train_data = data.iloc[max(0, idx - train_step):idx]
                #sys.exit("Stopping execution here")
                print(f" TRAINING START date: {train_data.index[0]}")
                print(f" TRAINING END date: {train_data.index[-1]}")
                print(f" (C)urrent (R)eturn (P)eroid: {crp}")
                print(f" (N)ext (R)eturn (P)eroid {nrp}")
                
                # Current training period return
                returns, r_i, sigma = StatisticCalculatorRollingD(train_data, 
                                                                  train_start=train_data.index[0],
                                                                  train_end=train_data.index[-1],
                                                                  freq = train_freq, 
                                                                  print_out = False, 
                                                                  plot = False)
                
                # Get the closest past available trading day
                rebalance_date = pd.Timestamp(crp)
                latest_prices = np.array(data.loc[rebalance_date].values) # Current month lastest price
                
                # Fetch the adjust closed price on rebalance day and calcualte 
                # how to rebalance our portfolio

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
                qp_H = portfoilo_Hamiltonian.to_quadratic_program_k(k=k) # return a quadratic program
                #qp_H = portfoilo_Hamiltonian.to_quadratic_program() # return a quadratic program
                #print(qp_H)
                # Classic solver 
                classical_algorithm = NumPyMinimumEigensolver()
                classical_eigensolver = MinimumEigenOptimizer(classical_algorithm)
                print(" --- Solving: result_Classic --- ")
                result_Classic = classical_eigensolver.solve(qp_H)

                # Store results in a dictionary
                results_data.append({
                    "Date": crp.strftime("%Y-%m-%d"),
                    "Objective Value": result_Classic.fval,
                    "Optimal Selection": list(result_Classic.x)
                })

                latest_prices_zip = dict(zip(tickers, latest_prices))

                print(f"before_optimized_current_value:{budget}")

                share_allocation_classic = decode_results_rolling_k(result=result_Classic, 
                                                                tickers=tickers, 
                                                                num_bits = num_bits, 
                                                                crp=crp,
                                                                latest_prices = latest_prices_zip,
                                                                k = k)
                
                allocation_results.append(share_allocation_classic)
                val_crp = sum(share_allocation_classic[ticker] * latest_prices_zip[ticker] for ticker in data)
                print(f"crp: {crp}")
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
                                (portfolio_value[-2] + new_investment))
                                /(portfolio_value[-2] + new_investment))
                    return_.append(ret) # this is the return on the rebalance date before adding extra fundings

                # === Calculate our portfolio value for the next period when we use allocation_results
                # The next month is the first invest month, therefore, we will start put our money in it.
                if nrp:
                    while nrp not in all_data[tickers].index:
                        nrp += BDay(1)  # Move to the next business day if necessary
                    new_latest_prices = np.array(data.loc[nrp].values)

                # Next period ticker price
                new_latest_prices_zip = dict(zip(tickers, new_latest_prices))

                # Value on next month based on allocation from previous month at the sametime
                val_nrp = sum(share_allocation_classic[ticker] * new_latest_prices_zip[ticker] for ticker in data)

                # Update our budget
                budget = val_nrp + new_investment
                
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
            invested_amounts = [init_budget + i * new_investment for i in range(len(df_optimized_portfolio))]

            # Convert to Series aligned with index
            invested_series = pd.Series(invested_amounts, index=df_optimized_portfolio.index)
            #print(f"invested_series: {invested_series}")
            # Cumulative return accounting for new capital
            df_portfolio_returns = ((df_optimized_portfolio.squeeze() - invested_series) / invested_series) * 100

            # ===================== Save dataframe ===================== #

            # Save data
            filename = f"portfolio_returns_exp_return{expected_returns_rate}_lambda1_{lambda1}_lambda2_{lambda2}.csv"
            df_portfolio_returns.to_csv(os.path.join(results_dir, filename), index=False)

            # Save difference data 
            df_cost_diff = pd.DataFrame(cost_diff, index=crps)
            filename = f"cost_diff_{expected_returns_rate}_lambda1_{lambda1}_lambda2_{lambda2}.csv"
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
            filename = f"allocation_{expected_returns_rate}_lambda1_{lambda1}_lambda2_{lambda2}.csv"
            allocation_results.to_csv(os.path.join(results_dir, filename), index=False)

# Save execution times as a CSV for tracking
df_exec_times = pd.DataFrame(execution_times, columns=["Lambda1", "Lambda2", "ExpectedReturnRate", "ExecutionTime"])
df_exec_times.to_csv(os.path.join(results_dir, "execution_times.csv"), index=False)

print(f"df_portfolio_returns:\n {df_portfolio_returns}")
print(f"df_benchmarks_returns:\n {df_benchmarks_returns}")

""""""
# ===================== Plotting ===================== #
# Plot optimized portfolio vs benchmarks
plt.figure(figsize=(12, 6))
plt.plot(df_portfolio_returns, label="Optimized Portfolio", linewidth=2, color='black')

for ticker in bm_tickers:
    if ticker in target_tickers:
        plt.plot(df_benchmarks_returns[ticker], label=ticker, linewidth=2, linestyle = "-")
    else:
        plt.plot(df_benchmarks_returns[ticker], label=ticker, linestyle="--")

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
