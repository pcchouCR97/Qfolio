"""
Simplified results export - auto-extract metrics from objects instead of 30+ params
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

def save_backtest_results(
    portfolio_history: list,
    port_stats: dict,
    spy_stats: dict,
    opt_config,  # OptimizationConfig object
    run_config: dict,
    output_dir: str = 'results',
    include_timeseries: bool = False,
    advanced_metrics: Optional[dict] = None,
    filename: Optional[str] = None
) -> str:
    """
    Save backtest results to JSON with auto-extraction from objects.

    Args:
        portfolio_history: List of daily portfolio records
        port_stats: Dict from calculate_performance_stats()
        spy_stats: Dict from bm.get_statistics()['SPY']
        opt_config: OptimizationConfig object
        run_config: Dict with sim config {sim_start_date, sim_end_date, initial_budget, ...}
        output_dir: Directory to save results
        include_timeseries: Include full time series data
        advanced_metrics: Optional dict with QuantStats metrics (alpha, beta, sortino, etc.)
        filename: Custom filename (auto-generated if None)

    Returns:
        Path to saved file

    Example run_config:
        {
            'sim_start_date': '2020-07-01',
            'sim_end_date': '2025-10-28',
            'initial_budget': 18000,
            'new_invest_per_period': 0,
            'training_lookback_days': 120,
            'rebalance_freq': '63B',
            'exchange': 'NYSE',
            'sharpe_n': 3
        }

    Example advanced_metrics (for grid search):
        {
            'alpha': opt_alpha,
            'beta': opt_beta,
            'r_squared': opt_r2,
            'treynor_ratio': opt_treynor,
            'sortino_ratio': opt_sortino,
            'calmar_ratio': opt_calmar,
            'information_ratio': opt_information_ratio,
            'omega_ratio': opt_omega,
            'profit_factor': opt_profit_factor,
            'gain_to_pain_ratio': opt_gain_to_pain,
            'recovery_factor': opt_recovery_factor,
            'payoff_ratio': opt_payoff,
            'tail_ratio': opt_tail_ratio,
            'skewness': opt_skew,
            'kurtosis': opt_kurtosis,
            'upside_capture': upside_capture,
            'downside_capture': downside_capture,
            'capture_ratio': capture_ratio
        }
    """

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Auto-generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"backtest_{opt_config.solver_type}_k{opt_config.k}_lambda{opt_config.lambda1:.0e}_{timestamp}.json"

    # Build results dictionary
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'run_config': run_config
        },
        'optimization_config': {
            'k': opt_config.k,
            'lambda1': opt_config.lambda1,
            'q': opt_config.q,
            'H_scale': opt_config.H_scale,
            'solver_type': opt_config.solver_type
        },
        'performance': {
            'portfolio': {
                'total_return_pct': float(port_stats['Total Return (%)']),
                'annualized_return_pct': float(port_stats['Annualized Return (%)']),
                'volatility_pct': float(port_stats['Volatility (%)']),
                'sharpe_ratio': float(port_stats['Sharpe Ratio']),
                'max_drawdown_pct': float(port_stats['Max Drawdown (%)']),
                'win_rate_pct': float(port_stats['Win Rate (%)']),
                'total_days': int(port_stats['Total Days'])
            },
            'spy_benchmark': {
                'total_return_pct': float(spy_stats['Total Return (%)']),
                'annualized_return_pct': float(spy_stats['Annualized Return (%)']),
                'volatility_pct': float(spy_stats['Volatility (%)']),
                'sharpe_ratio': float(spy_stats['Sharpe Ratio']),
                'max_drawdown_pct': float(spy_stats['Max Drawdown (%)']),
                'win_rate_pct': float(spy_stats['Win Rate (%)']),
                'total_days': int(spy_stats['Total Days'])
            },
            'vs_benchmark': {
                'return_diff_pct': float(port_stats['Total Return (%)'] - spy_stats['Total Return (%)']),
                'sharpe_diff': float(port_stats['Sharpe Ratio'] - spy_stats['Sharpe Ratio']),
                'volatility_diff_pct': float(port_stats['Volatility (%)'] - spy_stats['Volatility (%)']),
            }
        }
    }

    # Add advanced metrics if provided (for grid search)
    if advanced_metrics:
        results['performance']['advanced'] = {
            k: float(v) if isinstance(v, (int, float)) else v
            for k, v in advanced_metrics.items()
        }

    # Optionally include time series data
    if include_timeseries:
        import pandas as pd

        dates = [record['date'].strftime('%Y-%m-%d') if isinstance(record['date'], pd.Timestamp) else str(record['date'])
                 for record in portfolio_history]
        values = [record['value'] for record in portfolio_history]

        results['timeseries'] = {
            'dates': dates,
            'portfolio_values': values
        }

    # Save to JSON
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_path}")
    return output_path
