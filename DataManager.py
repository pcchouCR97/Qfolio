import os
import glob
import datetime
import pandas as pd
import yfinance as yf
from pandas.tseries.offsets import BDay
from pandas.tseries.frequencies import to_offset

class DataManager:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

    def load_data(self, path=None):
        """
        Load price data from a CSV file.

        Parameters:
        - path (str, optional): Specific file path to load. If not provided,
            the method searches for the most recent 'all_data_*.csv' in the current directory.

        Returns:
        - pd.DataFrame: Loaded price data with datetime index, or None if nothing is found.
        """
        if path and os.path.exists(path):
            print(f"Loading data from: {path}")
            self.data = pd.read_csv(path, index_col=0, parse_dates=True)
            return self.data

        pattern = os.path.join(".", "all_data_*.csv")
        files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        if files:
            print(f"Loading latest data from: {files[0]}")
            self.data = pd.read_csv(files[0], index_col=0, parse_dates=True)
            return self.data

        print("No CSV file found in current directory.")
        return None

    def fetch_yahoo_finance_data(self, price = 'Close', fetch=False, save_to_file=False):
        """
        Fetch Yahoo Finance Data and optionally save it as a timestamped CSV.
        """
        if not fetch:
            print("Fetch is disabled. Skipping data download.")
            return None

        data = yf.download(self.tickers, start=self.start_date, end=self.end_date)[price]

        if save_to_file:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"all_data_{timestamp}_{'_'.join(self.tickers)}_{self.start_date}to{self.end_date}.csv"
            data.to_csv(filename)
            print(f"Stock data saved as: {filename}")
        else:
            print("Data fetched but not saved.")

        return data

    def get_valid_dates(self, start, end, freq='21B'):
        """
        Get valid rebalance dates within a range using a frequency.
        """
        offset = to_offset(freq)
        idx = self.data.loc[start:end].index
        step = offset.n if hasattr(offset, 'n') else 1
        return idx[::step], idx[step::step]

    def adjust_to_business_day(self, date):
        """
        Adjust a date forward to the next valid business day in the dataset.
        """
        date = pd.Timestamp(date)
        while date not in self.data.index:
            date += BDay(1)
        return date
