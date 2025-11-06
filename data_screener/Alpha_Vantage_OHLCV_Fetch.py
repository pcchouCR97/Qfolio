
import requests
import pandas as pd
import time
import os

class AlphaVantageOHLCVClient:
    """
    A client to fetch full OHLCV (Open, High, Low, Close, Volume) data from Alpha Vantage.
    Based on the user's reference Alpha_advantage_fetch.py script.
    """
    def __init__(self, api_key):
        if api_key == 'YOUR_API_KEY_HERE' or not api_key:
            raise ValueError("API key is not set. Please edit the script and provide a valid Alpha Vantage API key.")
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.requests_made_this_minute = 0
        self.minute_start_time = time.time()

    def _respect_rate_limits(self):
        """Pauses execution to respect the Alpha Vantage API rate limit (e.g., 5 calls/min)."""
        self.requests_made_this_minute += 1
        if self.requests_made_this_minute >= 5:
            elapsed_time = time.time() - self.minute_start_time
            if elapsed_time < 60:
                wait_time = 60 - elapsed_time
                print(f"\n⏳ Rate limit reached. Waiting for {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            # Reset counter for the new minute
            self.requests_made_this_minute = 0
            self.minute_start_time = time.time()

    def get_daily_ohlcv(self, symbol, outputsize='full'):
        """
        Get full OHLCV and Adjusted Close data for a symbol.
        outputsize: 'compact' (last 100 days) or 'full' (20+ years of data).
        """
        params = {
            'function': 'TIME_SERIES_DAILY_ADJUSTED',
            'symbol': symbol.upper(),
            'outputsize': outputsize,
            'apikey': self.api_key
        }
        
        print(f"\n📡 Fetching full OHLCV data for {symbol.upper()}...")
        self._respect_rate_limits()

        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if "Error Message" in data:
                print(f"❌ API Error for {symbol}: {data['Error Message']}")
                return None
            if "Note" in data:
                print(f"❌ Rate Limit Error for {symbol}: {data['Note']}")
                return None

            time_series_key = "Time Series (Daily)"
            if time_series_key not in data:
                print(f"❌ No time series data found for {symbol}")
                return None

            df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            # Rename columns to a standard, clean format
            df.rename(columns={
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. adjusted close': 'Adj Close',
                '6. volume': 'Volume'
            }, inplace=True)
            
            # Keep only the relevant columns and convert to numeric
            ohlcv_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            df = df[ohlcv_columns].apply(pd.to_numeric, errors='coerce')
            
            print(f"✅ Successfully fetched {len(df)} days of data for {symbol}")
            return df

        except Exception as e:
            print(f"❌ An error occurred for {symbol}: {e}")
            return None

def main():
    """Main execution function."""
    # -------------------------------------------------------------------
    # IMPORTANT: Replace with your actual Alpha Vantage API Key
    API_KEY = "API_KEY"
    # -------------------------------------------------------------------

    # Define the list of symbols you want to fetch
    symbols = ['AAPL', 'MSFT', 'AMZN', 'PLTR', 'ORCL', 'HOOD', 'VOO', 'SH']

    print("🚀 Starting Alpha Vantage OHLCV Fetcher")
    client = AlphaVantageOHLCVClient(api_key=API_KEY)
    
    all_data = {}
    failed_symbols = []

    for symbol in symbols:
        ticker_data = client.get_daily_ohlcv(symbol, outputsize='full')
        if ticker_data is not None and not ticker_data.empty:
            all_data[symbol] = ticker_data
        else:
            failed_symbols.append(symbol)

    if not all_data:
        print("\n❌ No data was downloaded. Exiting.")
        return

    # --- Proper Storage Method ---
    # Combine all DataFrames into a single DataFrame with multi-level columns
    print("\n⚙️ Combining all ticker data into a single DataFrame...")
    final_df = pd.concat(all_data, axis=1, keys=all_data.keys())
    # Swap column levels to have (e.g., 'Open', 'AAPL') instead of ('AAPL', 'Open')
    final_df = final_df.swaplevel(0, 1, axis=1).sort_index(axis=1)
    final_df.index.name = 'Date'

    # Save the combined data to a single CSV file
    output_filename = "all_stocks_ohlcv.csv"
    final_df.to_csv(output_filename)
    print(f"\n💾 Success! All data properly stored in: {output_filename}")
    print(f"Shape: {final_df.shape[0]} rows, {final_df.shape[1]} columns")

    if failed_symbols:
        print(f"\n❌ Failed to fetch data for: {failed_symbols}")

if __name__ == "__main__":
    main()
