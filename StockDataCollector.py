
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import yfinance as yf

class StockDataCollector():
    def __init__(
            self,
            tickers: list,
            start_date: str,
            end_date: str,
            price: str,
            save_to_database = True,
            fetch =True,
        ) -> None:
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.price = price
        self.save_to_database = save_to_database
        self.fetch = fetch

    def FetchYahooFinanceData(self):
        """
        Fetch Yahoo Finance Data
        """ 
        if self.fetch == True:
            data = yf.download(self.tickers, start=self.start_date, end=self.end_date)[self.price]

            if self.save_to_database == False:
                print(f"{data}")
                print("!! Data is not saved.")

            else:
                # Get current date and time
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"+ "_".join(self.tickers) + self.start_date + "to" + self.end_date)

                # Save data with timestamp
                filename = f"all_data_{timestamp}.csv"
                data.to_csv(filename)

                print(f"Stock data saved as: {filename} locally!")
                print(data.head())  # Preview first few rows
        
        elif self.fetch == False:
            print("Please provide database if any")
            return None
        
        return data