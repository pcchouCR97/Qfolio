"""
Results Loading and Analysis Helper Functions

Utilities for loading, processing, and analyzing backtest results from grid search.
"""

import json
import os
import glob
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy import stats


def load_all_results(results_dir: str = "results") -> pd.DataFrame:
    """
    Load all JSON result files into a single DataFrame.

    Parameters:
    -----------
    results_dir : str
        Directory containing result JSON files

    Returns:
    --------
    pd.DataFrame
        Combined results with all metrics
    """
    json_files = glob.glob(os.path.join(results_dir, "results_*.json"))

    if not json_files:
        raise FileNotFoundError(f"No result files found in {results_dir}/")

    data = []

    for filepath in json_files:
        with open(filepath, 'r') as f:
            result = json.load(f)

        # Flatten nested structure
        row = {
            # Metadata
            'method': result['metadata']['method'],
            'solver_type': result['metadata']['solver_type'],
            'q': result['metadata']['q'],
            'budget': result['metadata']['budget'],
            'lambda1': result['metadata']['lambda1'],
            'k': result['metadata']['k'],
            'H_scale': result['metadata']['H_scale'],

            # Performance
            'total_return_pct': result['performance']['total_return_pct'],
            'cagr_pct': result['performance']['cagr_pct'],
            'sharpe_ratio': result['performance']['sharpe_ratio'],
            'sortino_ratio': result['performance']['sortino_ratio'],
            'calmar_ratio': result['performance']['calmar_ratio'],
            'treynor_ratio': result['performance']['treynor_ratio'],
            'information_ratio': result['performance']['information_ratio'],
            'max_drawdown_pct': result['performance']['max_drawdown_pct'],
            'volatility_pct': result['performance']['volatility_pct'],
            'alpha_pct': result['performance']['alpha_pct'],
            'beta': result['performance']['beta'],
            'r_squared': result['performance']['r_squared'],
            'omega_ratio': result['performance']['omega_ratio'],
            'profit_factor': result['performance']['profit_factor'],
            'gain_to_pain': result['performance']['gain_to_pain'],
            'recovery_factor': result['performance']['recovery_factor'],
            'payoff_ratio': result['performance']['payoff_ratio'],
            'tail_ratio': result['performance']['tail_ratio'],
            'win_rate_pct': result['performance']['win_rate_pct'],
            'skewness': result['performance']['skewness'],
            'kurtosis': result['performance']['kurtosis'],

            # Market capture
            'upside_capture_pct': result['market_capture']['upside_capture_pct'],
            'downside_capture_pct': result['market_capture']['downside_capture_pct'],
            'capture_ratio': result['market_capture']['capture_ratio'],

            # Portfolio characteristics
            'avg_num_assets': result['portfolio_characteristics']['avg_num_assets'],
            'avg_turnover_pct': result['portfolio_characteristics']['avg_turnover_pct'],
            'num_rebalances': result['portfolio_characteristics']['num_rebalances'],
            'avg_sector_hhi': result['portfolio_characteristics']['avg_sector_hhi'],
            'avg_sectors_held': result['portfolio_characteristics']['avg_sectors_held'],
            'max_single_asset_weight_pct': result['portfolio_characteristics']['max_single_asset_weight_pct'],
            'avg_max_asset_weight_pct': result['portfolio_characteristics']['avg_max_asset_weight_pct'],

            # Benchmark comparison
            'spy_total_return_pct': result['benchmark_comparison']['spy_total_return_pct'],
            'excess_return_pct': result['benchmark_comparison']['excess_return_pct'],
            'outperformance_pct': result['benchmark_comparison']['outperformance_pct'],
            'correlation': result['benchmark_comparison']['correlation'],
            'tracking_error_pct': result['benchmark_comparison']['tracking_error_pct'],

            # Statistical tests
            't_statistic': result['statistical_tests']['t_statistic'],
            'p_value': result['statistical_tests']['p_value'],
            'significance_level': result['statistical_tests']['significance_level'],

            # Drawdown
            'max_drawdown_duration_days': result['drawdown_analysis']['max_drawdown_duration_days'],
            'avg_drawdown_duration_days': result['drawdown_analysis']['avg_drawdown_duration_days'],

            # Filepath for reference
            'filepath': filepath
        }

        data.append(row)

    df = pd.DataFrame(data)

    # Sort by method, solver, q, budget
    df = df.sort_values(['method', 'solver_type', 'q', 'budget']).reset_index(drop=True)

    print(f"[OK] Loaded {len(df)} backtest results")
    print(f"  - Methods: {df['method'].unique()}")
    print(f"  - Solvers: {df['solver_type'].unique()}")
    print(f"  - q range: {df['q'].min():.0e} to {df['q'].max():.0e}")
    print(f"  - Budget range: ${df['budget'].min():.0f} to ${df['budget'].max():.0f}")

    return df


def calculate_consistency_score(df: pd.DataFrame) -> pd.Series:
    """
    Calculate consistency score: Sharpe × (1 - MaxDD/100) × sqrt(WinRate/50)

    This metric balances:
    - Risk-adjusted returns (Sharpe)
    - Capital preservation (MaxDD)
    - Reliability (Win Rate)

    Parameters:
    -----------
    df : pd.DataFrame
        Results DataFrame with sharpe_ratio, max_drawdown_pct, win_rate_pct

    Returns:
    --------
    pd.Series
        Consistency scores
    """
    sharpe = df['sharpe_ratio']
    max_dd = df['max_drawdown_pct'].abs()  # Make positive
    win_rate = df['win_rate_pct']

    # Consistency = Sharpe × (1 - MaxDD/100) × sqrt(WinRate/50)
    consistency = sharpe * (1 - max_dd/100) * np.sqrt(win_rate/50)

    return consistency


def generate_comparison_matrix(df: pd.DataFrame,
                               metric: str = 'sharpe_ratio') -> pd.DataFrame:
    """
    Generate head-to-head comparison matrix for Classical vs QAOA.

    Parameters:
    -----------
    df : pd.DataFrame
        Results DataFrame
    metric : str
        Metric to compare (e.g., 'sharpe_ratio', 'calmar_ratio')

    Returns:
    --------
    pd.DataFrame
        Comparison matrix with columns: q, budget, classical, qaoa, delta, winner
    """
    # Separate by solver type
    classical = df[df['solver_type'] == 'classic'].copy()
    qaoa = df[df['solver_type'] == 'QAOA'].copy()

    # Merge on (q, budget)
    comparison = pd.merge(
        classical[['q', 'budget', metric]],
        qaoa[['q', 'budget', metric]],
        on=['q', 'budget'],
        suffixes=('_classical', '_qaoa'),
        how='inner'
    )

    # Calculate delta and winner
    comparison['delta'] = comparison[f'{metric}_qaoa'] - comparison[f'{metric}_classical']
    comparison['delta_pct'] = (comparison['delta'] / comparison[f'{metric}_classical'].abs()) * 100
    comparison['winner'] = comparison['delta'].apply(lambda x: 'QAOA' if x > 0 else 'Classical' if x < 0 else 'Tie')

    return comparison


def statistical_comparison(df: pd.DataFrame,
                           metric: str = 'sharpe_ratio') -> Dict:
    """
    Perform statistical tests comparing Classical vs QAOA.

    Uses paired t-test since we're comparing same (q, budget) configurations.

    Parameters:
    -----------
    df : pd.DataFrame
        Results DataFrame
    metric : str
        Metric to test

    Returns:
    --------
    Dict
        Statistical test results including t-stat, p-value, effect size
    """
    comparison = generate_comparison_matrix(df, metric)

    classical_values = comparison[f'{metric}_classical'].values
    qaoa_values = comparison[f'{metric}_qaoa'].values

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(qaoa_values, classical_values)

    # Effect size (Cohen's d for paired samples)
    diff = qaoa_values - classical_values
    effect_size = np.mean(diff) / np.std(diff, ddof=1)

    # Mean advantage
    mean_classical = np.mean(classical_values)
    mean_qaoa = np.mean(qaoa_values)
    mean_delta = mean_qaoa - mean_classical
    mean_delta_pct = (mean_delta / abs(mean_classical)) * 100 if mean_classical != 0 else 0

    # Win rate
    qaoa_wins = (qaoa_values > classical_values).sum()
    total_comparisons = len(qaoa_values)
    win_rate = (qaoa_wins / total_comparisons) * 100

    results = {
        'metric': metric,
        't_statistic': t_stat,
        'p_value': p_value,
        'effect_size': effect_size,
        'mean_classical': mean_classical,
        'mean_qaoa': mean_qaoa,
        'mean_delta': mean_delta,
        'mean_delta_pct': mean_delta_pct,
        'qaoa_wins': qaoa_wins,
        'classical_wins': total_comparisons - qaoa_wins,
        'win_rate_pct': win_rate,
        'n_comparisons': total_comparisons,
        'significant': p_value < 0.05,
        'highly_significant': p_value < 0.01
    }

    return results


def get_top_configs(df: pd.DataFrame,
                   n: int = 5,
                   metric: str = 'sharpe_ratio',
                   by_solver: bool = True) -> pd.DataFrame:
    """
    Get top N configurations by specified metric.

    Parameters:
    -----------
    df : pd.DataFrame
        Results DataFrame
    n : int
        Number of top configs to return
    metric : str
        Metric to rank by
    by_solver : bool
        If True, get top N per solver type

    Returns:
    --------
    pd.DataFrame
        Top configurations
    """
    # Add consistency score if using it
    if metric == 'consistency_score':
        df = df.copy()
        df['consistency_score'] = calculate_consistency_score(df)

    if by_solver:
        top_configs = []
        for solver in df['solver_type'].unique():
            solver_df = df[df['solver_type'] == solver]
            # Handle negative metrics (like max_drawdown)
            if 'drawdown' in metric.lower():
                top = solver_df.nlargest(n, metric)  # Less negative is better
            else:
                top = solver_df.nlargest(n, metric)
            top_configs.append(top)
        result = pd.concat(top_configs, ignore_index=True)
    else:
        if 'drawdown' in metric.lower():
            result = df.nlargest(n, metric)
        else:
            result = df.nlargest(n, metric)

    return result


def create_pivot_table(df: pd.DataFrame,
                      metric: str = 'sharpe_ratio',
                      solver: str = 'classic') -> pd.DataFrame:
    """
    Create pivot table (q × Budget) for heatmap visualization.

    Parameters:
    -----------
    df : pd.DataFrame
        Results DataFrame
    metric : str
        Metric to pivot
    solver : str
        Solver type to filter by ('classic' or 'QAOA')

    Returns:
    --------
    pd.DataFrame
        Pivot table with q as rows, budget as columns
    """
    filtered = df[df['solver_type'] == solver].copy()

    # Add consistency score if needed
    if metric == 'consistency_score':
        filtered['consistency_score'] = calculate_consistency_score(filtered)

    pivot = filtered.pivot_table(
        index='q',
        columns='budget',
        values=metric,
        aggfunc='mean'
    )

    # Sort index and columns
    pivot = pivot.sort_index(ascending=True)
    pivot = pivot.sort_index(axis=1, ascending=True)

    return pivot


def identify_pareto_frontier(df: pd.DataFrame,
                             x_metric: str = 'max_drawdown_pct',
                             y_metric: str = 'sharpe_ratio') -> pd.DataFrame:
    """
    Identify Pareto-optimal configurations.

    A configuration is Pareto-optimal if no other configuration is strictly better
    on both metrics.

    Parameters:
    -----------
    df : pd.DataFrame
        Results DataFrame
    x_metric : str
        X-axis metric (typically risk, e.g., max_drawdown_pct)
    y_metric : str
        Y-axis metric (typically return, e.g., sharpe_ratio)

    Returns:
    --------
    pd.DataFrame
        Pareto-optimal configurations marked with 'is_pareto' column
    """
    df = df.copy()
    df['is_pareto'] = False

    # Convert drawdown to positive for comparison (less negative is better)
    x_values = -df[x_metric].values if 'drawdown' in x_metric.lower() else df[x_metric].values
    y_values = df[y_metric].values

    n = len(df)
    for i in range(n):
        is_dominated = False
        for j in range(n):
            if i == j:
                continue
            # Check if j dominates i (better on both metrics)
            if x_values[j] >= x_values[i] and y_values[j] >= y_values[i]:
                if x_values[j] > x_values[i] or y_values[j] > y_values[i]:
                    is_dominated = True
                    break

        if not is_dominated:
            df.iloc[i, df.columns.get_loc('is_pareto')] = True

    return df


if __name__ == '__main__':
    """
    Test the helper functions.
    """
    print("="*90)
    print("Testing Results Helper Functions")
    print("="*90 + "\n")

    # Load results
    df = load_all_results()

    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df[['method', 'solver_type', 'q', 'budget', 'sharpe_ratio', 'max_drawdown_pct']].head(10))

    # Calculate consistency scores
    print("\n" + "="*90)
    print("Consistency Scores (Top 5)")
    print("="*90)
    df['consistency_score'] = calculate_consistency_score(df)
    top5 = get_top_configs(df, n=5, metric='consistency_score', by_solver=False)
    print(top5[['method', 'solver_type', 'q', 'budget', 'consistency_score']])

    # Statistical comparison
    print("\n" + "="*90)
    print("Statistical Comparison: Sharpe Ratio")
    print("="*90)
    stats_sharpe = statistical_comparison(df, 'sharpe_ratio')
    for key, value in stats_sharpe.items():
        print(f"{key:25s}: {value}")

    print("\n[OK] All tests completed successfully!")
