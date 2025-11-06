"""
Comprehensive Results Analysis and Visualization

Generates publication-ready figures, tables, and statistical summaries
from Classical vs QAOA grid search results.

Usage:
    python analyze_results.py

Outputs:
    - figures/*.png (heatmaps, line plots, Pareto frontiers)
    - tables/*.tex (LaTeX tables for paper)
    - analysis_summary.txt (text report)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from qfolio.analysis.load_results_helper import (
    load_all_results,
    calculate_consistency_score,
    generate_comparison_matrix,
    statistical_comparison,
    get_top_configs,
    create_pivot_table,
    identify_pareto_frontier
)

# Configuration
RESULTS_DIR = "results"
FIGURES_DIR = "figures"
TABLES_DIR = "tables"
DPI = 300
FIGSIZE_HEATMAP = (10, 8)
FIGSIZE_LINE = (12, 6)
CMAP_DIVERGING = 'RdYlGn'  # Red-Yellow-Green for differences
CMAP_SEQUENTIAL = 'viridis'  # For single metrics

# Create output directories
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)


def generate_heatmap(pivot: pd.DataFrame,
                     title: str,
                     filename: str,
                     cmap: str = CMAP_SEQUENTIAL,
                     fmt: str = '.2f',
                     vmin: float = None,
                     vmax: float = None,
                     center: float = None):
    """Generate a single heatmap with proper formatting."""
    fig, ax = plt.subplots(figsize=FIGSIZE_HEATMAP)

    # Create heatmap
    sns.heatmap(
        pivot,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        center=center,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': title.split('(')[0].strip()},
        linewidths=0.5,
        linecolor='gray',
        ax=ax
    )

    # Format axes
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Budget ($)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Risk Aversion (q)', fontsize=12, fontweight='bold')

    # Format budget labels
    budget_labels = [f"${int(b/1000)}k" if b < 1000000 else f"${int(b/1000000)}M"
                     for b in pivot.columns]
    ax.set_xticklabels(budget_labels, rotation=45, ha='right')

    # Format q labels
    q_labels = [f"{q:.0e}" for q in pivot.index]
    ax.set_yticklabels(q_labels, rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=DPI, bbox_inches='tight')
    plt.close()

    print(f"[OK] Saved: {filename}")


def generate_all_heatmaps(df: pd.DataFrame):
    """Generate all comparison heatmaps."""
    print("\n" + "="*90)
    print("GENERATING HEATMAPS")
    print("="*90)

    metrics = {
        'sharpe_ratio': ('Sharpe Ratio', '.2f'),
        'max_drawdown_pct': ('Max Drawdown (%)', '.1f'),
        'calmar_ratio': ('Calmar Ratio', '.2f'),
        'cagr_pct': ('CAGR (%)', '.1f'),
        'avg_turnover_pct': ('Avg Turnover (%)', '.0f'),
        'max_single_asset_weight_pct': ('Max Concentration (%)', '.1f'),
        'consistency_score': ('Consistency Score', '.2f')
    }

    # Add consistency score
    df['consistency_score'] = calculate_consistency_score(df)

    for metric, (metric_name, fmt) in metrics.items():
        # Classical heatmap (Regular + Classic)
        pivot_classical = create_pivot_table(df, metric, 'classic')
        generate_heatmap(
            pivot_classical,
            f"{metric_name} - Classical (Regular + NumPy)",
            f"heatmap_{metric}_classical.png",
            cmap=CMAP_SEQUENTIAL,
            fmt=fmt
        )

        # QAOA heatmap (Hybrid + QAOA + Refinement)
        pivot_qaoa = create_pivot_table(df, metric, 'QAOA')
        generate_heatmap(
            pivot_qaoa,
            f"{metric_name} - QAOA Hybrid (QAOA + Refinement)",
            f"heatmap_{metric}_qaoa.png",
            cmap=CMAP_SEQUENTIAL,
            fmt=fmt
        )

        # Difference heatmap (QAOA+Ref - Classical)
        pivot_diff = pivot_qaoa - pivot_classical
        generate_heatmap(
            pivot_diff,
            f"{metric_name} - Difference (QAOA+Ref - Classical)",
            f"heatmap_{metric}_delta.png",
            cmap=CMAP_DIVERGING,
            fmt=fmt,
            center=0  # Center colormap at zero
        )

    print(f"\n[OK] Generated {len(metrics) * 3} heatmaps")


def generate_line_plots(df: pd.DataFrame):
    """Generate line plots showing trends."""
    print("\n" + "="*90)
    print("GENERATING LINE PLOTS")
    print("="*90)

    # 1. Sharpe vs q (for selected budgets)
    fig, ax = plt.subplots(figsize=FIGSIZE_LINE)

    budgets_to_plot = [10000, 50000, 100000]
    for budget in budgets_to_plot:
        # Classical
        classical_data = df[(df['solver_type'] == 'classic') & (df['budget'] == budget)]
        ax.plot(classical_data['q'], classical_data['sharpe_ratio'],
                '--o', label=f'Classical ${int(budget/1000)}k', alpha=0.7)

        # QAOA
        qaoa_data = df[(df['solver_type'] == 'QAOA') & (df['budget'] == budget)]
        ax.plot(qaoa_data['q'], qaoa_data['sharpe_ratio'],
                '-o', label=f'QAOA ${int(budget/1000)}k', linewidth=2)

    ax.set_xscale('log')
    ax.set_xlabel('Risk Aversion (q)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
    ax.set_title('Sharpe Ratio vs Risk Aversion (Selected Budgets)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'lineplot_sharpe_vs_q.png'), dpi=DPI, bbox_inches='tight')
    plt.close()

    print("[OK] Saved: lineplot_sharpe_vs_q.png")

    # 2. Sharpe vs Budget (for selected q values)
    fig, ax = plt.subplots(figsize=FIGSIZE_LINE)

    q_values_to_plot = [1e-4, 1e-3, 1e-2]
    for q in q_values_to_plot:
        # Classical
        classical_data = df[(df['solver_type'] == 'classic') & (df['q'] == q)]
        ax.plot(classical_data['budget'], classical_data['sharpe_ratio'],
                '--o', label=f'Classical q={q:.0e}', alpha=0.7)

        # QAOA
        qaoa_data = df[(df['solver_type'] == 'QAOA') & (df['q'] == q)]
        ax.plot(qaoa_data['budget'], qaoa_data['sharpe_ratio'],
                '-o', label=f'QAOA q={q:.0e}', linewidth=2)

    ax.set_xscale('log')
    ax.set_xlabel('Budget ($)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
    ax.set_title('Sharpe Ratio vs Budget (Selected Risk Aversions)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'lineplot_sharpe_vs_budget.png'), dpi=DPI, bbox_inches='tight')
    plt.close()

    print("[OK] Saved: lineplot_sharpe_vs_budget.png")

    # 3. Win Rate Bar Chart
    metrics = ['sharpe_ratio', 'calmar_ratio', 'cagr_pct', 'max_drawdown_pct']
    metric_names = ['Sharpe', 'Calmar', 'CAGR', 'MaxDD (lower)']
    win_rates = []

    for metric in metrics:
        comparison = generate_comparison_matrix(df, metric)
        if 'drawdown' in metric.lower():
            # For drawdown, less negative is better (QAOA wins if delta > 0)
            qaoa_wins = (comparison['delta'] > 0).sum()
        else:
            qaoa_wins = (comparison['delta'] > 0).sum()
        total = len(comparison)
        win_rates.append((qaoa_wins / total) * 100)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metric_names, win_rates, color=['green' if wr > 50 else 'red' for wr in win_rates], alpha=0.7)

    ax.axhline(50, color='black', linestyle='--', linewidth=1, label='50% (Tie)')
    ax.set_ylabel('QAOA Win Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('QAOA vs Classical: Win Rate by Metric', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend()

    # Add value labels on bars
    for bar, wr in zip(bars, win_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{wr:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'barchart_win_rates.png'), dpi=DPI, bbox_inches='tight')
    plt.close()

    print("[OK] Saved: barchart_win_rates.png")

    print(f"\n[OK] Generated 3 line/bar plots")


def generate_pareto_frontier(df: pd.DataFrame):
    """Generate Pareto frontier plot."""
    print("\n" + "="*90)
    print("GENERATING PARETO FRONTIER")
    print("="*90)

    fig, ax = plt.subplots(figsize=FIGSIZE_LINE)

    # Identify Pareto-optimal configs for each solver
    for solver, marker, color in [('classic', 'o', 'blue'), ('QAOA', 's', 'red')]:
        solver_df = df[df['solver_type'] == solver].copy()
        solver_df = identify_pareto_frontier(solver_df, 'max_drawdown_pct', 'sharpe_ratio')

        # Plot all points
        non_pareto = solver_df[~solver_df['is_pareto']]
        pareto = solver_df[solver_df['is_pareto']]

        ax.scatter(non_pareto['max_drawdown_pct'], non_pareto['sharpe_ratio'],
                   marker=marker, s=50, alpha=0.3, color=color)

        ax.scatter(pareto['max_drawdown_pct'], pareto['sharpe_ratio'],
                   marker=marker, s=150, alpha=0.8, color=color,
                   edgecolors='black', linewidths=2,
                   label=f'{solver.upper()} Pareto-Optimal')

    ax.set_xlabel('Max Drawdown (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
    ax.set_title('Efficient Frontier: Sharpe Ratio vs Max Drawdown', fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'pareto_frontier.png'), dpi=DPI, bbox_inches='tight')
    plt.close()

    print("[OK] Saved: pareto_frontier.png")


def generate_latex_tables(df: pd.DataFrame):
    """Generate LaTeX tables for paper."""
    print("\n" + "="*90)
    print("GENERATING LATEX TABLES")
    print("="*90)

    # Add consistency score
    df['consistency_score'] = calculate_consistency_score(df)

    # Table 1: Top 5 Configurations by Consistency Score
    top5 = get_top_configs(df, n=5, metric='consistency_score', by_solver=True)

    latex_top5 = """\\begin{table}[htbp]
\\centering
\\caption{Top 5 Configurations by Consistency Score}
\\label{tab:top5_configs}
\\begin{tabular}{lccccccc}
\\toprule
Method & Solver & q & Budget & Sharpe & MaxDD (\\%) & Calmar & Consistency \\\\
\\midrule
"""

    for _, row in top5.iterrows():
        latex_top5 += f"{row['method']} & {row['solver_type']} & {row['q']:.0e} & \\${row['budget']:.0f} & "
        latex_top5 += f"{row['sharpe_ratio']:.2f} & {row['max_drawdown_pct']:.1f} & "
        latex_top5 += f"{row['calmar_ratio']:.2f} & {row['consistency_score']:.2f} \\\\\n"

    latex_top5 += """\\bottomrule
\\end{tabular}
\\end{table}
"""

    with open(os.path.join(TABLES_DIR, 'table_top5_configs.tex'), 'w') as f:
        f.write(latex_top5)

    print("[OK] Saved: table_top5_configs.tex")

    # Table 2: Statistical Comparison
    metrics = ['sharpe_ratio', 'calmar_ratio', 'cagr_pct', 'max_drawdown_pct', 'avg_turnover_pct']
    metric_names = ['Sharpe Ratio', 'Calmar Ratio', 'CAGR (\\%)', 'Max Drawdown (\\%)', 'Turnover (\\%)']

    latex_stats = """\\begin{table}[htbp]
\\centering
\\caption{Statistical Comparison: Classical vs QAOA (Paired t-test)}
\\label{tab:statistical_comparison}
\\begin{tabular}{lcccccc}
\\toprule
Metric & Classical & QAOA & $\\Delta$ & $\\Delta$ (\\%) & p-value & Sig. \\\\
\\midrule
"""

    for metric, metric_name in zip(metrics, metric_names):
        stats = statistical_comparison(df, metric)
        sig_marker = '***' if stats['p_value'] < 0.001 else '**' if stats['p_value'] < 0.01 else '*' if stats['p_value'] < 0.05 else ''

        latex_stats += f"{metric_name} & {stats['mean_classical']:.2f} & {stats['mean_qaoa']:.2f} & "
        latex_stats += f"{stats['mean_delta']:.2f} & {stats['mean_delta_pct']:.1f}\\% & "
        latex_stats += f"{stats['p_value']:.3f} & {sig_marker} \\\\\n"

    latex_stats += """\\bottomrule
\\multicolumn{7}{l}{\\footnotesize *** p<0.001, ** p<0.01, * p<0.05} \\\\
\\end{tabular}
\\end{table}
"""

    with open(os.path.join(TABLES_DIR, 'table_statistical_comparison.tex'), 'w') as f:
        f.write(latex_stats)

    print("[OK] Saved: table_statistical_comparison.tex")

    print(f"\n[OK] Generated 2 LaTeX tables")


def generate_text_summary(df: pd.DataFrame):
    """Generate comprehensive text summary."""
    print("\n" + "="*90)
    print("GENERATING TEXT SUMMARY")
    print("="*90)

    summary = []
    summary.append("="*90)
    summary.append("COMPREHENSIVE RESULTS ANALYSIS: Classical vs QAOA+Refinement")
    summary.append("="*90)
    summary.append("")
    summary.append("IMPORTANT NOTE:")
    summary.append("-"*90)
    summary.append("This comparison is between:")
    summary.append("  1. Classical (Regular method + NumPy exact solver)")
    summary.append("  2. QAOA Hybrid (Hybrid method + QAOA solver + Classical refinement)")
    summary.append("")
    summary.append("The QAOA Hybrid includes an additional classical refinement stage.")
    summary.append("Performance differences reflect BOTH the quantum solver AND the refinement step.")
    summary.append("For isolated solver comparison, run additional experiments:")
    summary.append("  - Regular + QAOA (pure quantum)")
    summary.append("  - Hybrid + Classical (classical + refinement)")
    summary.append("")

    # Dataset overview
    summary.append("DATASET OVERVIEW")
    summary.append("-"*90)
    summary.append(f"Total backtests: {len(df)}")
    summary.append(f"  - Classical: {len(df[df['solver_type'] == 'classic'])}")
    summary.append(f"  - QAOA: {len(df[df['solver_type'] == 'QAOA'])}")
    summary.append(f"Grid search dimensions:")
    summary.append(f"  - q values: {sorted(df['q'].unique())}")
    summary.append(f"  - Budgets: {sorted(df['budget'].unique())}")
    summary.append("")

    # Statistical comparison for key metrics
    summary.append("STATISTICAL COMPARISON (Paired t-tests)")
    summary.append("-"*90)

    metrics = ['sharpe_ratio', 'calmar_ratio', 'cagr_pct', 'max_drawdown_pct', 'avg_turnover_pct', 'max_single_asset_weight_pct']
    metric_names = ['Sharpe Ratio', 'Calmar Ratio', 'CAGR (%)', 'Max Drawdown (%)', 'Turnover (%)', 'Max Concentration (%)']

    for metric, metric_name in zip(metrics, metric_names):
        stats = statistical_comparison(df, metric)

        summary.append(f"\n{metric_name}:")
        summary.append(f"  Classical mean: {stats['mean_classical']:.2f}")
        summary.append(f"  QAOA mean:      {stats['mean_qaoa']:.2f}")
        summary.append(f"  Difference:     {stats['mean_delta']:.2f} ({stats['mean_delta_pct']:+.1f}%)")
        summary.append(f"  t-statistic:    {stats['t_statistic']:.3f}")
        summary.append(f"  p-value:        {stats['p_value']:.4f} {'***' if stats['highly_significant'] else '**' if stats['significant'] else ''}")
        summary.append(f"  QAOA wins:      {stats['qaoa_wins']}/{stats['n_comparisons']} ({stats['win_rate_pct']:.1f}%)")

    summary.append("")

    # Best configurations
    summary.append("BEST CONFIGURATIONS")
    summary.append("-"*90)

    df['consistency_score'] = calculate_consistency_score(df)

    summary.append("\nBy Consistency Score:")
    for solver in ['classic', 'QAOA']:
        best = df[df['solver_type'] == solver].nlargest(1, 'consistency_score').iloc[0]
        summary.append(f"\n{solver.upper()}:")
        summary.append(f"  q={best['q']:.0e}, Budget=${best['budget']:.0f}")
        summary.append(f"  Sharpe: {best['sharpe_ratio']:.2f}, MaxDD: {best['max_drawdown_pct']:.1f}%, Calmar: {best['calmar_ratio']:.2f}")
        summary.append(f"  Consistency Score: {best['consistency_score']:.2f}")

    summary.append("\nBy Sharpe Ratio:")
    for solver in ['classic', 'QAOA']:
        best = df[df['solver_type'] == solver].nlargest(1, 'sharpe_ratio').iloc[0]
        summary.append(f"\n{solver.upper()}:")
        summary.append(f"  q={best['q']:.0e}, Budget=${best['budget']:.0f}")
        summary.append(f"  Sharpe: {best['sharpe_ratio']:.2f}, MaxDD: {best['max_drawdown_pct']:.1f}%, Calmar: {best['calmar_ratio']:.2f}")

    summary.append("")

    # Overall conclusion
    summary.append("OVERALL CONCLUSION")
    summary.append("-"*90)

    sharpe_stats = statistical_comparison(df, 'sharpe_ratio')
    winner = "QAOA" if sharpe_stats['mean_qaoa'] > sharpe_stats['mean_classical'] else "Classical"

    summary.append(f"\nWinner: {winner}")
    summary.append(f"  QAOA achieves {sharpe_stats['mean_qaoa']:.2f} Sharpe vs Classical {sharpe_stats['mean_classical']:.2f}")
    summary.append(f"  Improvement: {sharpe_stats['mean_delta_pct']:+.1f}% (p={sharpe_stats['p_value']:.4f})")
    summary.append(f"  QAOA wins {sharpe_stats['qaoa_wins']}/{sharpe_stats['n_comparisons']} head-to-head comparisons ({sharpe_stats['win_rate_pct']:.1f}%)")

    summary.append("")
    summary.append("="*90)

    # Write to file
    summary_text = "\n".join(summary)
    with open('analysis_summary.txt', 'w') as f:
        f.write(summary_text)

    print(summary_text)
    print("\n[OK] Saved: analysis_summary.txt")


def main():
    """Run complete analysis."""
    print("\n" + "="*90)
    print("COMPREHENSIVE RESULTS ANALYSIS")
    print("="*90)

    # Load results
    df = load_all_results(RESULTS_DIR)

    # Filter out q=0.2 outlier (incomplete grid coverage)
    print(f"\nFiltering data: Removing q=0.2 outlier...")
    df_before = len(df)
    df = df[df['q'] < 0.2].copy()
    df_after = len(df)
    print(f"  Removed {df_before - df_after} rows with q>=0.2")
    print(f"  Remaining: {df_after} backtests")

    # Generate all outputs
    generate_all_heatmaps(df)
    generate_line_plots(df)
    generate_pareto_frontier(df)
    generate_latex_tables(df)
    generate_text_summary(df)

    print("\n" + "="*90)
    print("ANALYSIS COMPLETE!")
    print("="*90)
    print(f"\nOutputs:")
    print(f"  - Figures:  {FIGURES_DIR}/")
    print(f"  - Tables:   {TABLES_DIR}/")
    print(f"  - Summary:  analysis_summary.txt")
    print("\n")


if __name__ == '__main__':
    main()
