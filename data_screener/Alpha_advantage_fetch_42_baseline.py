import requests
import pandas as pd
import time
import json
from datetime import datetime, timedelta
import os
from data_list import return_all_symbols
import glob

class AlphaVantageClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.requests_made = 0
        self.max_requests_per_day = 75 # Purchased plan limit
        self.max_requests_per_minute = 5
        self.last_request_time = 0
        
    def _respect_rate_limits(self):
        """Add delays to respect rate limits"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        # Ensure at least 12 seconds between requests (5 per minute)
        min_interval = 12
        if time_since_last_request < min_interval:
            wait_time = min_interval - time_since_last_request
            print(f"⏳ Waiting {wait_time:.1f} seconds for rate limit...")
            time.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    def _make_request(self, params):
        """Make API request with error handling"""
        if self.requests_made >= self.max_requests_per_day:
            print(f"❌ Daily limit reached ({self.max_requests_per_day} requests)")
            return None
        
        self._respect_rate_limits()
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            self.requests_made += 1
            
            print(f"📊 Request #{self.requests_made}/{self.max_requests_per_day}")
            
            # Check for API errors
            if "Error Message" in data:
                print(f"❌ API Error: {data['Error Message']}")
                return None
            elif "Note" in data:
                print(f"⚠️ Rate Limited: {data['Note']}")
                print("💡 Try again in a few minutes or tomorrow")
                return None
            elif "Information" in data:
                print(f"ℹ️ API Info: {data['Information']}")
                return None
            
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error: {e}")
            return None
    
    def get_adjusted_close(self, symbol, outputsize='full'):
        """
        Get ADJUSTED CLOSE price data for a symbol
        outputsize: 'compact' (last 100 days) or 'full' (20+ years of data)
        Returns only the adjusted close column
        """
        params = {
            'function': 'TIME_SERIES_DAILY_ADJUSTED',
            'symbol': symbol.upper(),
            'outputsize': outputsize,
            'apikey': self.api_key
        }
        
        print(f"\n📡 Fetching adjusted close for {symbol.upper()}...")
        data = self._make_request(params)
        
        if data is None:
            return None
        
        # Extract time series data
        time_series_key = "Time Series (Daily)"
        if time_series_key not in data:
            print(f"❌ No data found for {symbol}")
            return None
        
        return self._extract_adjusted_close(data[time_series_key], symbol)
    
    def _extract_adjusted_close(self, time_series_data, symbol):
        """Extract only the adjusted close price from time series data"""
        df = pd.DataFrame.from_dict(time_series_data, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Extract only adjusted close (column '5. adjusted close')
        if '5. adjusted close' in df.columns:
            adj_close = pd.to_numeric(df['5. adjusted close'], errors='coerce')
            adj_close = adj_close.to_frame()
            adj_close.columns = [symbol]
        else:
            print(f"❌ Adjusted close not found for {symbol}")
            return None
        
        print(f"✅ Successfully fetched {len(adj_close)} days of adjusted close for {symbol}")
        return adj_close

# Main execution
def main():
    # Replace with your actual API key
    API_KEY = "YOUR API"
    
    # Initialize client
    client = AlphaVantageClient(API_KEY)
    
    # Symbols to fetch
    # symbols = return_all_symbols()
    # Reference: https://finance.yahoo.com/quote/SPY/
    # REference: SPY: https://finance.yahoo.com/sectors/consumer-cyclical/

    # ==========================================
    # All stocks symbol are extracted from S&P 500 list as of 2025-10-18
    # Selected 41 stocks across all 11 GICS sectors + SPY as benchmark = 42 total.
    # Ranked by Market Cap for each sector.
    symbols = [
    # Technology: 6 stocks (NVDA, MSFT, AAPL, AVGO, ORCL, PLTR)
    'NVDA', 'MSFT', 'AAPL', 'AVGO', 'ORCL', 'PLTR',
    # Financials: 6 stocks (BRK.B, JPM, V, BAC, MA, GS)
    'BRK.B', 'JPM', 'V', 'BAC', 'MA','GS',
    # Healthcare: 5 stocks (LLY, JNJ, ABBV, UNH, ABT)
    'LLY', 'JNJ', 'ABBV', 'UNH', 'ABT',
    # Consumer Cyclical: 4 stocks (AMZN, TSLA, HD, MCD)
    'AMZN', 'TSLA', 'HD', 'MCD',
    # Industrials: 4 stocks (GE, CAT, RTX, GEV)
    'GE', 'CAT', 'RTX', 'GEV',
    # Consumer Defensive: 4 stocks ( WMT, COST, PG, KO)
    'WMT', 'COST', 'PG', 'KO',
    # Communication Services: 3 stocks (GOOGL, META, NFLX)
    'GOOGL', 'META', 'NFLX',
    # Energy: 3 stocks (XOM, CVX, COP)
    'XOM', 'CVX', 'COP',
    # Materials: 2 stocks (LIN, SCCO)
    'LIN', 'SCCO',
    # Utilities: 2 stocks (NEE, CEG)
    'NEE', 'CEG',
    # Real Estate: 2 stocks (WELL, PLD)
    'WELL', 'PLD',
    # Benchmark
    'SPY'    # S&P 500 ETF
    ]
    # ==========================================
    

    print("🚀 Starting Alpha Vantage Adjusted Close Fetch")
    print(f"📋 Symbols: {', '.join(symbols)}")
    print(f"🔑 Daily limit: {client.max_requests_per_day} requests")
    print(f"⚠️  Warning: You have {len(symbols)} symbols but only {client.max_requests_per_day} requests/day")
    print("="*60)
    
    # Store all adjusted close prices in one DataFrame
    all_adj_close = pd.DataFrame()
    failed_symbols = []
    
    for i, symbol in enumerate(symbols):
        print(f"\n{'='*20} {symbol} ({i+1}/{len(symbols)}) {'='*20}")
        
        # Check if we're about to exceed daily limit
        if client.requests_made >= client.max_requests_per_day:
            print(f"\n⚠️  Daily limit reached! Remaining symbols will be skipped.")
            failed_symbols.extend(symbols[i:])
            break
        
        # Get adjusted close data
        adj_close = client.get_adjusted_close(symbol, outputsize='full')
        
        if adj_close is not None:
            # Add to master DataFrame
            all_adj_close = pd.concat([all_adj_close, adj_close], axis=1)
            
            # Display summary
            print(f"\n📊 {symbol} Adjusted Close Summary:")
            print(f"Date Range: {adj_close.index[0].date()} to {adj_close.index[-1].date()}")
            print(f"Total Days: {len(adj_close)}")
            print(f"Latest Price: ${adj_close[symbol].iloc[-1]:.2f}")
            print(f"Highest Price: ${adj_close[symbol].max():.2f}")
            print(f"Lowest Price: ${adj_close[symbol].min():.2f}")
            
            # Calculate 1-year return if we have enough data
            if len(adj_close) >= 252:  # Trading days in a year
                one_year_ago_price = adj_close[symbol].iloc[-252]
                latest_price = adj_close[symbol].iloc[-1]
                returns = ((latest_price - one_year_ago_price) / one_year_ago_price) * 100
                print(f"1-Year Return: {returns:.2f}%")
            
            # Show recent prices
            print(f"\n📈 Last 5 days of adjusted close:")
            print(adj_close.tail())
            
        else:
            print(f"❌ Failed to fetch data for {symbol}")
            failed_symbols.append(symbol)
    
    # Save combined data to CSV
    if not all_adj_close.empty:
        filename = f"SP500_42stocks_baseline_adjusted_close.csv"
        all_adj_close.to_csv(filename)
        print(f"\n💾 Combined data saved to: {filename}")
        
        # Display final combined DataFrame summary
        print(f"\n{'='*60}")
        print("📊 COMBINED ADJUSTED CLOSE PRICES")
        print(f"{'='*60}")
        print(f"\nShape: {all_adj_close.shape[0]} days × {all_adj_close.shape[1]} stocks")
        print(f"\nLatest Prices:")
        print(all_adj_close.iloc[-1].to_string())
        
        print(f"\n\n📈 Recent Data (Last 10 days):")
        print(all_adj_close.tail(10))
        
    # Final summary
    print(f"\n{'='*60}")
    print("🎉 Data fetching completed!")
    print(f"{'='*60}")
    print(f"✅ Successfully fetched: {all_adj_close.columns.tolist()}")
    if failed_symbols:
        print(f"❌ Failed/Skipped: {failed_symbols}")
    print(f"📈 Total API requests made: {client.requests_made}/{client.max_requests_per_day}")
    print(f"🔄 Remaining requests today: {client.max_requests_per_day - client.requests_made}")
    
    return all_adj_close

if __name__ == "__main__":
    data = main()