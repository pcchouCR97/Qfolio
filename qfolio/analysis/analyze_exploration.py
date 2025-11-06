"""
Comprehensive Analysis of Parameter Space Exploration

Generates publication-quality heatmaps, line plots, and LaTeX tables for paper.

Usage:
    python analyze_exploration.py

Input:
    results/results_*.json (from grid search)

Output:
    figures/ - PNG charts (300 DPI)
    tables/ - LaTeX tables
    analysis_summary.txt - Text summary

Author: Quantum Portfolio Optimization Research Team
Date: 2025-10-19
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from save_results_module import extract_metrics_dataframe, calculate_consistency_score

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'

# ===============================================================================
# HEATMAP GENERATION
# ===============================================================================

def generate_heatmap(df, method, metric, title, filename, cmap='RdYlGn', fmt='.2f', vmin=None, vmax=None):
    """Generate a single heatmap for (q × Budget) parameter space."""

    # Filter to specific method
    method_df = df[df['method'] == method].copy()

    # Pivot table
    pivot = method_df.pivot(index='q', columns='budget', values=metric)

    # Sort indices
    pivot = pivot.sort_index(ascending=False)  # q from high to low (top to bottom)
    pivot = pivot.sort_index(axis=1, ascending=True)  # budget low to high (left to right)

    # Create heatmap
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap, center=None,
                vmin=vmin, vmax=vmax,
                linewidths=0.5, linecolor='gray',
                cbar_kws={'label': metric.replace('_', ' ').title()})

    plt.title(f'{title}\n{method.capitalize()} Method', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Budget ($)', fontsize=12)
    plt.ylabel('Risk Aversion (q)', fontsize=12)

    # Format axis labels
    plt.yticks(range(len(pivot.index)), [f'{q:.0e}' for q in pivot.index], rotation=0)
    plt.xticks(range(len(pivot.columns)), [f'${int(b):,}' for b in pivot.columns], rotation=45, ha='right')

    plt.tight_layout()

    # Save
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f'figures/{filename}', bbox_inches='tight')
    plt.close()

    print(f"✓ Heatmap saved: {filename}")

    return pivot


def generate_all_heatmaps(df):
    """Generate all 7 key heatmaps per method."""

    for method in ['regular', 'hybrid']:
        print(f"\nGenerating heatmaps for {method} method...")

        # 1. Sharpe Ratio (most important!)
        generate_heatmap(df, method, 'sharpe_ratio',
                        'Sharpe Ratio Across Parameter Space',
                        f'heatmap_{method}_sharpe.png', cmap='RdYlGn', fmt='.2f', vmin=0, vmax=3)

        # 2. Max Drawdown
        generate_heatmap(df, method, 'max_drawdown_pct',
                        'Maximum Drawdown (%)',
                        f'heatmap_{method}_maxdd.png', cmap='RdYlGn_r', fmt='.1f', vmin=-50, vmax=0)

        # 3. Calmar Ratio
        generate_heatmap(df, method, 'calmar_ratio',
                        'Calmar Ratio (Return / Max Drawdown)',
                        f'heatmap_{method}_calmar.png', cmap='RdYlGn', fmt='.2f', vmin=0, vmax=2)

        # 4. Consistency Score
        generate_heatmap(df, method, 'consistency_score',
                        'Consistency Score',
                        f'heatmap_{method}_consistency.png', cmap='viridis', fmt='.2f', vmin=0, vmax=2)

        # 5. Max Asset Weight (concentration)
        generate_heatmap(df, method, 'max_single_asset_weight_pct',
                        'Max Single-Asset Weight (%)',
                        f'heatmap_{method}_concentration.png', cmap='YlOrRd', fmt='.0f', vmin=0, vmax=100)

        # 6. Turnover
        generate_heatmap(df, method, 'avg_turnover_pct',
                        'Average Turnover per Rebalance (%)',
                        f'heatmap_{method}_turnover.png', cmap='Blues', fmt='.1f', vmin=0, vmax=100)

        # 7. CAGR
        generate_heatmap(df, method, 'cagr_pct',
                        'Compound Annual Growth Rate (%)',
                        f'heatmap_{method}_cagr.png', cmap='RdYlGn', fmt='.1f', vmin=0, vmax=50)


# ===============================================================================
# LINE PLOTS
# ===============================================================================

def plot_q_sensitivity(df, budget_levels=[10000, 50000, 100000]):
    """Plot Sharpe vs q at different budget levels."""

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, method in enumerate(['regular', 'hybrid']):
        ax = axes[idx]
        method_df = df[df['method'] == method]

        for budget in budget_levels:
            budget_data = method_df[method_df['budget'] == budget].sort_values('q')
            ax.plot(budget_data['q'], budget_data['sharpe_ratio'],
                   marker='o', linewidth=2, markersize=8,
                   label=f'${budget:,}', alpha=0.8)

        ax.set_title(f'{method.capitalize()} Method: Sharpe vs q', fontsize=12, fontweight='bold')
        ax.set_xlabel('Risk Aversion (q)', fontsize=10)
        ax.set_ylabel('Sharpe Ratio', fontsize=10)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(title='Budget', fontsize=9)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Sharpe=1')

    plt.suptitle('Risk Aversion Sensitivity Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/lineplot_q_sensitivity.png', bbox_inches='tight')
    plt.close()

    print("✓ Line plot saved: lineplot_q_sensitivity.png")


def plot_budget_scaling(df, q_levels=[1e-4, 1e-3, 1e-2]):
    """Plot ROI vs Budget at different q levels."""

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, method in enumerate(['regular', 'hybrid']):
        ax = axes[idx]
        method_df = df[df['method'] == method]

        for q in q_levels:
            q_data = method_df[method_df['q'] == q].sort_values('budget')
            ax.plot(q_data['budget'], q_data['total_return_pct'],
                   marker='o', linewidth=2, markersize=8,
                   label=f'q={q:.0e}', alpha=0.8)

        ax.set_title(f'{method.capitalize()} Method: ROI vs Budget', fontsize=12, fontweight='bold')
        ax.set_xlabel('Budget ($)', fontsize=10)
        ax.set_ylabel('Total Return (%)', fontsize=10)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(title='Risk Aversion', fontsize=9)

    plt.suptitle('Budget Scalability Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/lineplot_budget_scaling.png', bbox_inches='tight')
    plt.close()

    print("✓ Line plot saved: lineplot_budget_scaling.png")


def plot_pareto_frontier(df):
    """Plot Sharpe vs MaxDD (efficient frontier)."""

    plt.figure(figsize=(12, 8))

    for method in ['regular', 'hybrid']:
        method_df = df[df['method'] == method]
        plt.scatter(method_df['max_drawdown_pct'], method_df['sharpe_ratio'],
                   s=100, alpha=0.6, label=method.capitalize())

    plt.xlabel('Max Drawdown (%)', fontsize=12)
    plt.ylabel('Sharpe Ratio', fontsize=12)
    plt.title('Efficient Frontier: Sharpe vs Max Drawdown', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.axhline(y=1.5, color='red', linestyle='--', alpha=0.3, label='Target Sharpe=1.5')
    plt.axvline(x=-30, color='red', linestyle='--', alpha=0.3, label='MDD Limit=-30%')
    plt.tight_layout()
    plt.savefig('figures/pareto_frontier.png', bbox_inches='tight')
    plt.close()

    print("✓ Scatter plot saved: pareto_frontier.png")


# ===============================================================================
# LATEX TABLES
# ===============================================================================

def generate_top_configs_table(df, method, n=5):
    """Generate LaTeX table of top N configurations by consistency score."""

    method_df = df[df['method'] == method].copy()
    method_df = method_df.sort_values('consistency_score', ascending=False).head(n)

    table_df = method_df[['q', 'budget', 'sharpe_ratio', 'max_drawdown_pct',
                          'cagr_pct', 'max_single_asset_weight_pct', 'consistency_score']].copy()

    table_df['q'] = table_df['q'].apply(lambda x: f'${x:.0e}$')
    table_df['budget'] = table_df['budget'].apply(lambda x: f'\\${x:,.0f}')
    table_df.columns = ['$q$', 'Budget', 'Sharpe', 'Max DD (\%)', 'CAGR (\%)', 'Max Weight (\%)', 'Score']

    latex = table_df.to_latex(index=False, escape=False, float_format='%.2f',
                               caption=f'Top {n} configurations for {method} method (ranked by consistency score)',
                               label=f'tab:top_{method}')

    os.makedirs('tables', exist_ok=True)
    filepath = f'tables/table_top_{method}_configs.tex'
    with open(filepath, 'w') as f:
        f.write(latex)

    print(f"✓ Table saved: {filepath}")

    return table_df


# ===============================================================================
# TEXT SUMMARY
# ===============================================================================

def generate_text_summary(df):
    """Generate text summary of key findings."""

    summary = []
    summary.append("="*90)
    summary.append("PARAMETER SPACE EXPLORATION SUMMARY")
    summary.append("="*90)
    summary.append("")

    for method in ['regular', 'hybrid']:
        method_df = df[df['method'] == method]

        summary.append(f"\n{method.upper()} METHOD:")
        summary.append("-" * 50)

        # Best configuration
        best_idx = method_df['consistency_score'].idxmax()
        best = method_df.loc[best_idx]

        summary.append(f"Best Configuration:")
        summary.append(f"  q = {best['q']:.0e}")
        summary.append(f"  Budget = ${best['budget']:,.0f}")
        summary.append(f"  Sharpe = {best['sharpe_ratio']:.2f}")
        summary.append(f"  Max DD = {best['max_drawdown_pct']:.1f}%")
        summary.append(f"  CAGR = {best['cagr_pct']:.1f}%")
        summary.append(f"  Consistency Score = {best['consistency_score']:.2f}")
        summary.append("")

        # Best Sharpe
        best_sharpe_idx = method_df['sharpe_ratio'].idxmax()
        best_sharpe = method_df.loc[best_sharpe_idx]
        summary.append(f"Highest Sharpe:")
        summary.append(f"  q = {best_sharpe['q']:.0e}, Budget = ${best_sharpe['budget']:,.0f}")
        summary.append(f"  Sharpe = {best_sharpe['sharpe_ratio']:.2f}, Max DD = {best_sharpe['max_drawdown_pct']:.1f}%")
        summary.append("")

        # Lowest Max DD
        best_dd_idx = method_df['max_drawdown_pct'].idxmax()  # Least negative
        best_dd = method_df.loc[best_dd_idx]
        summary.append(f"Lowest Drawdown:")
        summary.append(f"  q = {best_dd['q']:.0e}, Budget = ${best_dd['budget']:,.0f}")
        summary.append(f"  Max DD = {best_dd['max_drawdown_pct']:.1f}%, Sharpe = {best_dd['sharpe_ratio']:.2f}")
        summary.append("")

    summary.append("="*90)

    summary_text = "\n".join(summary)

    # Save to file
    with open('analysis_summary.txt', 'w') as f:
        f.write(summary_text)

    print("\n" + summary_text)
    print("\n✓ Summary saved: analysis_summary.txt")


# ===============================================================================
# MAIN EXECUTION
# ===============================================================================

def main():
    """Main analysis workflow."""

    print("\n" + "="*90)
    print("PARAMETER SPACE EXPLORATION ANALYSIS")
    print("="*90 + "\n")

    # Load all results
    print("Loading results...")
    df = extract_metrics_dataframe('results')

    if df.empty:
        print("ERROR: No results found in 'results/' directory")
        print("Please run explore_regular.py and explore_hybrid.py first")
        return

    print(f"Loaded {len(df)} backtest results")
    print(f"Methods: {df['method'].unique()}")
    print(f"q values: {sorted(df['q'].unique())}")
    print(f"Budgets: {sorted(df['budget'].unique())}\n")

    # Generate all outputs
    print("\nGenerating heatmaps...")
    generate_all_heatmaps(df)

    print("\nGenerating line plots...")
    plot_q_sensitivity(df)
    plot_budget_scaling(df)
    plot_pareto_frontier(df)

    print("\nGenerating LaTeX tables...")
    for method in ['regular', 'hybrid']:
        generate_top_configs_table(df, method, n=5)

    print("\nGenerating text summary...")
    generate_text_summary(df)

    print("\n" + "="*90)
    print("ANALYSIS COMPLETE")
    print("="*90)
    print("Figures: figures/")
    print("Tables:  tables/")
    print("Summary: analysis_summary.txt")
    print("="*90 + "\n")


if __name__ == '__main__':
    main()
