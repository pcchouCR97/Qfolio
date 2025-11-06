"""
Check portfolio concentration across all rebalancing periods.

This script loads a result JSON file and examines how concentration
varies over time (across different rebalancing dates).
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def analyze_single_backtest(filepath):
    """Analyze concentration across time for a single backtest."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    metadata = data['metadata']
    performance = data['performance']

    # Check if portfolio_history exists
    if 'portfolio_history' not in data:
        print(f"[WARNING] No portfolio_history in {filepath}")
        return None

    history = data['portfolio_history']

    # Analyze each rebalancing period
    periods = []
    for period in history:
        if 'weights' not in period or 'rebalance_date' not in period:
            continue

        weights = period['weights']
        rebal_date = period['rebalance_date']

        # Calculate max weight
        if weights and isinstance(weights, dict):
            weight_values = list(weights.values())
            max_weight = max(weight_values) if weight_values else 0
            num_assets = len(weights)

            periods.append({
                'date': rebal_date,
                'max_weight_pct': max_weight * 100,
                'num_assets': num_assets,
                'weights': weights
            })

    if len(periods) == 0:
        print(f"[WARNING] No valid periods in {filepath}")
        return None

    return {
        'metadata': metadata,
        'periods': periods,
        'avg_max_weight': sum(p['max_weight_pct'] for p in periods) / len(periods),
        'overall_max_weight': performance.get('max_single_asset_weight_pct', 0)
    }


def main():
    """Analyze concentration across time for NumPy and QAOA results."""

    # Find representative examples
    numpy_files = glob.glob('results/results_regular_classic_*.json')
    qaoa_files = glob.glob('results/results_regular_QAOA_*.json')

    print("="*90)
    print("CONCENTRATION OVER TIME ANALYSIS")
    print("="*90)

    # Analyze a few representative configs
    configs_to_check = [
        ('q1e-03_B10000', 'Medium q, Medium budget'),
        ('q1e-04_B1000', 'Low q, Low budget'),
        ('q1e-02_B100000', 'High q, High budget'),
    ]

    for config_pattern, description in configs_to_check:
        print(f"\n\n{'='*90}")
        print(f"Configuration: {description} ({config_pattern})")
        print(f"{'='*90}")

        # Find matching files
        numpy_file = [f for f in numpy_files if config_pattern in f]
        qaoa_file = [f for f in qaoa_files if config_pattern in f]

        if not numpy_file or not qaoa_file:
            print(f"[WARNING] Missing files for {config_pattern}")
            continue

        # Analyze NumPy
        print(f"\n--- NumPy ---")
        numpy_result = analyze_single_backtest(numpy_file[0])
        if numpy_result:
            print(f"Overall max concentration: {numpy_result['overall_max_weight']:.1f}%")
            print(f"Average max concentration across periods: {numpy_result['avg_max_weight']:.1f}%")
            print(f"\nPeriod-by-period breakdown:")
            for i, period in enumerate(numpy_result['periods'][:5]):  # Show first 5
                print(f"  Period {i+1} ({period['date']}): {period['max_weight_pct']:.1f}% "
                      f"across {period['num_assets']} assets")
                # Show top holdings
                sorted_weights = sorted(period['weights'].items(),
                                       key=lambda x: x[1], reverse=True)
                for ticker, weight in sorted_weights[:3]:
                    print(f"    {ticker}: {weight*100:.1f}%")

        # Analyze QAOA
        print(f"\n--- QAOA ---")
        qaoa_result = analyze_single_backtest(qaoa_file[0])
        if qaoa_result:
            print(f"Overall max concentration: {qaoa_result['overall_max_weight']:.1f}%")
            print(f"Average max concentration across periods: {qaoa_result['avg_max_weight']:.1f}%")
            print(f"\nPeriod-by-period breakdown:")
            for i, period in enumerate(qaoa_result['periods'][:5]):  # Show first 5
                print(f"  Period {i+1} ({period['date']}): {period['max_weight_pct']:.1f}% "
                      f"across {period['num_assets']} assets")
                # Show top holdings
                sorted_weights = sorted(period['weights'].items(),
                                       key=lambda x: x[1], reverse=True)
                for ticker, weight in sorted_weights[:3]:
                    print(f"    {ticker}: {weight*100:.1f}%")

        # Create time series plot
        if numpy_result and qaoa_result:
            fig, ax = plt.subplots(figsize=(14, 6))

            numpy_dates = [p['date'] for p in numpy_result['periods']]
            numpy_max_weights = [p['max_weight_pct'] for p in numpy_result['periods']]

            qaoa_dates = [p['date'] for p in qaoa_result['periods']]
            qaoa_max_weights = [p['max_weight_pct'] for p in qaoa_result['periods']]

            ax.plot(numpy_dates, numpy_max_weights, marker='o', label='NumPy',
                   linewidth=2, markersize=8, alpha=0.7)
            ax.plot(qaoa_dates, qaoa_max_weights, marker='s', label='QAOA',
                   linewidth=2, markersize=8, alpha=0.7)

            ax.axhline(y=80, color='red', linestyle='--', linewidth=2,
                      label='Degenerate threshold (80%)')

            ax.set_xlabel('Rebalance Date', fontsize=12, fontweight='bold')
            ax.set_ylabel('Max Single Asset Weight (%)', fontsize=12, fontweight='bold')
            ax.set_title(f'Portfolio Concentration Over Time\n{description}',
                        fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=11)
            ax.grid(True, alpha=0.3)

            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            filename = f'concentration_over_time_{config_pattern}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"\n[OK] Saved: {filename}")


if __name__ == '__main__':
    main()
