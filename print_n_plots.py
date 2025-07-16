import numpy as np
import matplotlib.pyplot as plt
from qiskit.result import QuasiDistribution
import yfinance as yf
import pandas as pd

def print_result(result, model):
    """
    args: 
    result:

    model: class
    """
    selection = result.x
    value = result.fval
    print("Optimal: selection {}, value {:.4f}".format(selection, value))

    eigenstate = result.min_eigen_solver_result.eigenstate
    probabilities = (
        eigenstate.binary_probabilities()
        if isinstance(eigenstate, QuasiDistribution)
        else {k: np.abs(v) ** 2 for k, v in eigenstate.to_dict().items()}
    )
    print("\n----------------- Full result ---------------------")
    print("selection\tvalue\t\tprobability")
    print("---------------------------------------------------")
    probabilities = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

    for k, v in probabilities:
        x = np.array([int(i) for i in list(reversed(k))])
        value = model.to_quadratic_program().objective.evaluate(x)
        print("%10s\t%.4f\t\t%.4f" % (x, value, v))

def decode_results(result, tickers, num_bits):
    # Extract binary solution
    binary_solution = result.x  # Binary array from solver
    print(binary_solution)

    print(f"tickers = {tickers}")
    expected_size = len(tickers) * num_bits
    actual_size = len(binary_solution)
    print(f"Expected binary solution size: {expected_size}")
    print(f"Actual binary solution size: {actual_size}")

    print(f"{num_bits}")

    # Decode binary representation into number of shares
    share_allocation = {}
    for i, ticker in enumerate(tickers):
        share_value = sum(2**j * binary_solution[i * num_bits + j] for j in range(num_bits))
        share_allocation[ticker] = int(share_value)

    # Print final share allocation
    print("\n----- Portfolio Share Allocation -----")
    for ticker, shares in share_allocation.items():
        print(f"{ticker}: {shares} shares")
    
    return share_allocation

def decode_results_rolling(result, tickers, num_bits, crp, latest_prices):
    # Extract binary solution
    binary_solution = result.x  
    expected_size = len(tickers) * num_bits
    actual_size = len(binary_solution)

    #print(binary_solution)
    #print(f"tickers = {tickers}")
    print(f"Expected binary solution size: {expected_size}")
    print(f"Actual binary solution size: {actual_size} (should be the same as 'Expected binary solution size')")
    print(f"Number of binary bit used: {num_bits}")

    # Decode binary representation into number of shares
    share_allocation = {"Date": crp.strftime("%Y-%m-%d")}  # Add Month column
    total_value = {"Date": crp.strftime("%Y-%m-%d")}

    for i, ticker in enumerate(tickers):
        share_value = sum(2**j * binary_solution[i * num_bits + j] for j in range(num_bits))
        share_allocation[ticker] = int(share_value)

    # Print final share allocation
    print(f"\n----- Portfolio Share Allocation on {crp}-----")
    for ticker, shares in share_allocation.items():
        if ticker != "Date":  # Skip printing the month
            print(f"{ticker}: {shares} shares @ Price {latest_prices[ticker]}")
    
    return share_allocation  # Returns dictionary including the month


def decode_results_rolling_k(result, assets, num_bits, crp, latest_prices, k):
    # Extract binary solution
    binary_solution = result.x  
    expected_size = len(assets) * num_bits
    actual_size = len(binary_solution)

    #print(binary_solution)
    print(f"Expected binary solution size: {expected_size}")
    print(f"Actual binary solution size: {actual_size} (should be the same as 'Expected binary solution size')")
    print(f"Number of binary bit used: {num_bits}")

    # Decode binary representation into number of shares
    share_allocation = {"Date": crp.strftime("%Y-%m-%d")}  # Add Month column
    total_value = {"Date": crp.strftime("%Y-%m-%d")}

    for i, asset in enumerate(assets):
        share_value = sum(k**j * binary_solution[i * num_bits + j] for j in range(num_bits))
        share_allocation[asset] = int(share_value)

    # Print final share allocation
    print(f"\n----- Portfolio Share Allocation on {crp}-----")
    for asset, shares in share_allocation.items():
        if asset != "Date":  # Skip printing the month
            print(f"{asset}: {shares} shares @ Price {latest_prices[asset]}")
    
    return share_allocation  # Returns dictionary including the month

class track_back_plot():
    def __init__ (
            self,
            base_tickers: list,
            compare_tickers: list,
            start_date: str,
            end_date: str,
            share_allocation: dict, # the final result 
            data: pd.DataFrame, # 
            method: str,
            ) -> None:
        self.base_tickers = base_tickers
        self.compare_tickers = compare_tickers
        self.start_date = start_date
        self.end_date = end_date
        self.share_allocation = share_allocation
        self.data = data
        self.method = method

    def track_back_prep(self, download = False):
        # Define tickers and date range
        tickers_trackback = self.base_tickers
        start_date = self.start_date
        end_date = self.end_date

        if download == True:
            # Download historical adjusted closing prices
            self.data = yf.download(tickers_trackback, start=start_date, end=end_date)['Close']
        
        else:
            self.data = pd.read_csv('stock_data_2025-03-14_00-56-10.csv', index_col=0, parse_dates=True)
        
        if self.method == 'basic':
            # run plot
            self.PlotVsTarget()
        elif self.method == 'rolling':
            self.PlotVsTargetRolling()

        return None

    def PlotVsTargetRolling(self):
        """
        data: base data, required for future estimation

        """
        total_value = []
        for current_month in pd.date_range(self.start_date,self.end_date, freq= 'ME'):
            self.data.index = pd.to_datetime(self.data.index)  # Convert index to DateTimeIndex
            self.data = self.data.loc[self.start_date:current_month]  # Select date range

            #TODO shares * price / initial value( a return in time range)
            

            total = sum(self.data.loc[current_month,item] * self.share_allocation[item] for item in self.share_allocation)
            total_value.append(total)

            # Normalize prices to start at 100 for comparison
            normalized_prices = (total / total_value[0]) * 100
            print(f"Normalized price{normalized_prices}") 
        

        # Plotting
        plt.figure(figsize=(14, 7))
        for ticker in normalized_prices.columns:
            plt.plot(normalized_prices.index, normalized_prices[ticker], label=ticker)

        plt.plot(normalized_portfolio.index, normalized_portfolio, label='Opt Portfolio', linestyle='--')
        plt.title('Performance Comparison: AAPL, META, AMZN, NFLX, GOOGL, QQQ, and Custom-Weighted Portfolio\n(Jan 1, 2024 - Jan 1, 2025)')
        plt.xlabel('Date')
        plt.ylabel('Normalized Price (Starting at 100)')
        plt.legend()
        plt.grid(True)
        plt.show()

        return None

    def PlotVsTarget(self):
        data.index = pd.to_datetime(data.index)  # Convert index to DateTimeIndex
        data = data.loc[self.start_date:self.end_date]  # Select date range

        # Calculate monthly returns
        returns = data.resample('ME').last().pct_change().dropna()

        # Define custom portfolio weights
        base = sum(self.share_allocation[item] for item in self.share_allocation)
        weights = {ticker: self.share_allocation[ticker]/base for ticker in self.share_allocation}

        # Calculate weighted portfolio returns
        portfolio_returns = sum(returns[ticker] * weights[ticker] for ticker in self.share_allocation)

        # Normalize prices to start at 100 for comparison
        normalized_prices = (data / data.iloc[0]) * 100
        normalized_portfolio = (portfolio_returns + 1).cumprod() * 100

        # Plotting
        plt.figure(figsize=(14, 7))
        for ticker in normalized_prices.columns:
            plt.plot(normalized_prices.index, normalized_prices[ticker], label=ticker)

        #plt.plot(normalized_prices.index, normalized_prices['AAPL'], label='APPL')
        #plt.plot(normalized_prices.index, normalized_prices['AMZN'], label='AMZN')
        #plt.plot(normalized_prices.index, normalized_prices['META'], label='META')
        #plt.plot(normalized_prices.index, normalized_prices['QQQ'], label='QQQ')
        plt.plot(normalized_portfolio.index, normalized_portfolio, label='Opt Portfolio', linestyle='--')
        plt.title('Performance Comparison: AAPL, META, AMZN, NFLX, GOOGL, QQQ, and Custom-Weighted Portfolio\n(Jan 1, 2024 - Jan 1, 2025)')
        plt.xlabel('Date')
        plt.ylabel('Normalized Price (Starting at 100)')
        plt.legend()
        plt.grid(True)
        plt.show()
# test 
