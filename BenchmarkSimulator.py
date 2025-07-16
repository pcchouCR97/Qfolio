import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from pandas.tseries.frequencies import to_offset
import sys
import os
from utilities import save_files

class BenchmarkSimulator():
    def __init__(self, 
                 assets, 
                 data, 
                 init_budget, 
                 new_investment):
        self.assets = assets
        self.data = data
        self.init_budget = init_budget
        self.new_investment = new_investment

    def simulate(self, start_date, end_date, freq='21B', save = False, save_path = None):

        # simulate if all budget go into one asset
        # adjusting 1st market date due to holidays
        bms = pd.Timestamp(start_date)
        bme = pd.Timestamp(end_date)
        bms_adj = bms
        while bms_adj not in self.data[self.assets].index:
            bms_adj += BDay(1)

        bm_val = {a: [self.init_budget] for a in self.assets} # dict
        bm_shares = {a: [self.init_budget / self.data[a].loc[bms_adj]] for a in self.assets} # dict

        offset = to_offset(freq)
        step = offset.n if hasattr(offset, 'n') else 1
        # Get the list of all valid dates
        valid_dates = self.data.loc[bms:bme].index

        print(valid_dates)
        # Get crps (current rebalancing points) and nrps (next periods)
        crps = valid_dates[::step] # (c)urrent (r)eturn (p)eriod: current rebalance date
        nrps = valid_dates[step::step] # (n)ext (r)eturn (p)eriod: next rebalance date 
        print(crps)

        for crp, nrp in zip(crps, nrps):
            for a in self.assets:
                lv, ls = bm_val[a][-1], bm_shares[a][-1]
                bm_val[a].append(ls * self.data[a].loc[nrp] + self.new_investment)
                bm_shares[a].append(ls + self.new_investment / self.data[a].loc[nrp])

        df_benchmarks = pd.DataFrame(bm_val, index=crps)

        invested_amounts = [self.init_budget + i * self.new_investment for i in range(len(df_benchmarks))]

        # Convert to Series aligned with index
        invested_series = pd.Series(invested_amounts, index=df_benchmarks.index)

        # Step 2: Broadcast to match all columns (tickers)
        invested_df = pd.DataFrame({col: invested_series for col in df_benchmarks.columns})

        # Cumulative return accounting for new capital
        df_benchmarks_returns = df_benchmarks_returns = ((df_benchmarks - invested_df) / invested_df) * 100

        if save:
            # Ensure the directory exists
            os.makedirs(save_path, exist_ok=True)

            # Save Database
            df_benchmarks_returns.to_csv(os.path.join(save_path, "benchmarks_returns.csv"), index=False)

        return df_benchmarks_returns
    


# ==== unit test ==== # 

def test_benchmark_simulator():
    # === Load data from CSV ===
    csv_path = "all_data_2025-03-19_02-01-51AAPL_NVDA_MSFT_AMZN_VOO_XLF_XLE_XLV2018-01-01to2025-03-01.csv"  # <-- Replace with your actual file
    prices = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    assets = ['AAPL', 'MSFT']
    assert all(a in prices.columns for a in assets), "Missing required tickers in data"

    sim = BenchmarkSimulator(
        assets=assets,
        data=prices,
        init_budget=10000,
        new_investment=300
    )

    returns = sim.simulate(start_date='2025-01-01', end_date='2025-01-31', freq='5B')

    print("=== Benchmark Returns (%): ===")
    print(returns.round(2))

if __name__ == "__main__":
    test_benchmark_simulator()