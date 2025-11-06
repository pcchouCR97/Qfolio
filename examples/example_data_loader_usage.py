"""
Example: How to use qfolio/data_loader.py
"""

import pandas as pd
from qfolio.backtesting.date_loader import adjust_to_market_date, get_rebalance_dates, get_train_data

# Load full dataset
data = pd.read_csv('examples/example_csv/SP500_42stocks_baseline_adjusted_close_103025.csv', index_col=0, parse_dates=True)

print("="*80)
print("EXAMPLE 1: Get training data (fixed 252-day window)")
print("="*80)

# OLD WAY (from test_42_JP_QUANTUM_NO_CARD.py):
# prices_data = data.iloc[-252:]  # Just gets last 252 rows

# NEW WAY (with data_loader):
prices_data = get_train_data(
    data,
    train_end_date='2024-10-28',  # If this date is closed, auto-adjusts backward
    lookback='252B'  # Get exactly last 252 trading days
)

print(f"Training window: {prices_data.index[0]} to {prices_data.index[-1]}")
print(f"Number of days: {len(prices_data)}")
print()

# Calculate returns as usual
returns = prices_data.pct_change(fill_method=None).dropna()
expected_returns = returns.mean().values
covariances = returns.cov().values
current_prices = prices_data.iloc[-1].values

print(f"Expected returns shape: {expected_returns.shape}")
print(f"Covariances shape: {covariances.shape}")
print()

# ============================================================================
print("="*80)
print("EXAMPLE 2: Check if a specific date exists (weekend/holiday handling)")
print("="*80)

# Say you want to rebalance on 2024-01-01 (New Year's Day - market closed)
target_date = '2024-01-01'
adjusted_date = adjust_to_market_date(target_date, data, direction='forward')

print(f"Requested date: {target_date}")
print(f"Adjusted to (forward): {adjusted_date}")
print()

# Or adjust backward
adjusted_date_back = adjust_to_market_date(target_date, data, direction='backward')
print(f"Adjusted to (backward): {adjusted_date_back}")
print()

# ============================================================================
print("="*80)
print("EXAMPLE 3: Generate rebalance dates (every 1 month)")
print("="*80)

rebal_dates = get_rebalance_dates(
    start_date='2024-01-01',
    end_date='2024-10-28',
    data=data,
    freq='1M'  # Rebalance every 1 month
)

print(f"Generated {len(rebal_dates)} rebalance dates:")
for date in rebal_dates:
    print(f"  {date.strftime('%Y-%m-%d')}")
print()

# ============================================================================
print("="*80)
print("EXAMPLE 4: Generate rebalance dates (every 63 business days)")
print("="*80)

rebal_dates_63B = get_rebalance_dates(
    start_date='2024-01-01',
    end_date='2024-10-28',
    data=data,
    freq='63B'  # Rebalance every 63 trading days (~1 quarter)
)

print(f"Generated {len(rebal_dates_63B)} rebalance dates:")
for date in rebal_dates_63B:
    print(f"  {date.strftime('%Y-%m-%d')}")
print()

# ============================================================================
print("="*80)
print("EXAMPLE 5: Backtest loop with rebalancing")
print("="*80)

# Generate rebalance dates every 3 months
rebal_dates_3M = get_rebalance_dates(
    start_date='2023-01-01',
    end_date='2024-10-28',
    data=data,
    freq='3M'
)

print(f"Backtesting with {len(rebal_dates_3M)} rebalance periods:\n")

for i, rebal_date in enumerate(rebal_dates_3M):
    # Get training data: last 252 days before each rebalance date
    train_data = get_train_data(data, rebal_date, lookback='252B')

    # Calculate returns for this period
    returns = train_data.pct_change(fill_method=None).dropna()
    import numpy as np
    avg_return = float(np.mean(returns.mean()))

    print(f"Rebalance {i+1} on {rebal_date.strftime('%Y-%m-%d')}:")
    print(f"  Training: {train_data.index[0].strftime('%Y-%m-%d')} to {train_data.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Mean return: {avg_return:.6f}")

    # Here you would run your optimizer and get portfolio weights
    # optimizer.solve(...)

    print()

print("="*80)
print("INTEGRATION SUMMARY")
print("="*80)
print("To use in test_42_JP_QUANTUM_NO_CARD.py:")
print()
print("1. Replace lines 36-44:")
print("   OLD: prices_data = data.iloc[-252:]")
print("   NEW: from qfolio.data_loader import get_train_data")
print("        prices_data = get_train_data(data, '2024-10-28', '252B')")
print()
print("2. For backtesting, add:")
print("   from qfolio.data_loader import get_rebalance_dates")
print("   rebal_dates = get_rebalance_dates('2023-01-01', '2024-10-28', data, '1M')")
print()
print("3. For date checking:")
print("   from qfolio.data_loader import adjust_to_market_date")
print("   adjusted = adjust_to_market_date('2024-01-01', data, 'forward')")
print("="*80)
