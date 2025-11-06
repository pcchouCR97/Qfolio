# Qfolio Package - Data Loader

Simple data loader for portfolio optimization. No expanding windows, no rolling windows - just simple date handling and fixed lookback periods.

## Installation

The package is already in your project:
```
qfolio_package/
├── __init__.py
└── data_loader.py
```

## Quick Start

```python
from qfolio_package.data_loader import get_train_data

# Load your data
data = pd.read_csv('SP500_42stocks_baseline_adjusted_close_102825.csv',
                   index_col=0, parse_dates=True)

# Get last 252 trading days before 2024-10-28
prices_data = get_train_data(data, train_end_date='2024-10-28', lookback='252B')
```

## Functions

### 1. `get_train_data(data, train_end_date, lookback='252B')`

Get FIXED window of last X trading days. Just counts back X indices.

**Parameters:**
- `data`: DataFrame with DatetimeIndex
- `train_end_date`: End date (auto-adjusts if market closed)
- `lookback`: Window size (e.g., '252B', '63B', '21B')

**Example:**
```python
# Get last 1 year (252 trading days)
prices = get_train_data(data, '2024-10-28', '252B')

# Get last 1 quarter (63 trading days)
prices = get_train_data(data, '2024-10-28', '63B')
```

### 2. `adjust_to_market_date(date, data, direction='forward')`

If date is closed (weekend/holiday), find nearest trading day.

**Parameters:**
- `date`: Target date
- `data`: DataFrame with DatetimeIndex
- `direction`: 'forward' (default) or 'backward'

**Example:**
```python
# 2024-01-01 is New Year's Day (closed)
adjusted = adjust_to_market_date('2024-01-01', data, direction='forward')
# Returns: 2024-01-02 (first trading day)
```

### 3. `get_rebalance_dates(start_date, end_date, data, freq='1M')`

Generate rebalance dates at regular intervals.

**Parameters:**
- `start_date`: Start date
- `end_date`: End date
- `data`: DataFrame with DatetimeIndex
- `freq`: Frequency - '1M', '2M', '3M' (months) OR '21B', '63B', '252B' (business days)

**Example:**
```python
# Rebalance every 3 months
dates = get_rebalance_dates('2023-01-01', '2024-10-28', data, freq='3M')

# Rebalance every 63 business days (~1 quarter)
dates = get_rebalance_dates('2023-01-01', '2024-10-28', data, freq='63B')
```

## Integration Examples

### Replace existing code in test scripts

**OLD:**
```python
prices_data = data.iloc[-252:]
```

**NEW:**
```python
from qfolio_package.data_loader import get_train_data
prices_data = get_train_data(data, train_end_date='2024-10-28', lookback='252B')
```

### Backtesting loop

```python
from qfolio_package.data_loader import get_train_data, get_rebalance_dates

# Generate rebalance dates every 3 months
rebal_dates = get_rebalance_dates('2023-01-01', '2024-10-28', data, freq='3M')

# Loop over rebalance dates
for rebal_date in rebal_dates:
    # Get training data: last 252 days before each rebalance
    train_data = get_train_data(data, rebal_date, lookback='252B')

    # Calculate returns
    returns = train_data.pct_change(fill_method=None).dropna()
    expected_returns = returns.mean().values
    covariances = returns.cov().values

    # Run optimizer...
```

## Key Features

1. **Simple**: Just counts back indices - no complex date calculations
2. **Fixed windows**: No expanding, no rolling - exactly what you asked for
3. **Auto-adjusts**: Handles weekends/holidays automatically
4. **Flexible**: Easy to change window sizes (21B, 63B, 252B, etc.)

## Run Example

See `example_data_loader_usage.py` for complete working examples:
```bash
python example_data_loader_usage.py
```
