from pandas.tseries.offsets import BDay

# --- Dynamic Universe Helper Function ---
def get_eligible_stocks(current_date, all_stocks, data, lookback_days=240):
    """
    Returns stocks with sufficient historical data as of current_date.

    Following MSCI index methodology: stocks are eligible for selection once they
    have sufficient trading history for the required lookback window.

    Args:
    
        current_date (pd.Timestamp): The date at which to evaluate eligibility
        all_stocks (list): Full list of potential stocks to evaluate
        data (pd.DataFrame): Price data with stocks as columns
        lookback_days (int): Minimum number of business days of history required

    Returns:
        eligible (list): Stock tickers that have sufficient history
    """
    
    min_required_date = current_date - BDay(lookback_days)
    first_valid_dates = data[all_stocks].apply(lambda x: x.first_valid_index())
    eligible = first_valid_dates[first_valid_dates <= min_required_date].index.tolist()
    return eligible