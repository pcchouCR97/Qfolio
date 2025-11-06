"""
NumPy vs QAOA Comparison Analysis (Regular Methods Only - No GD)

Compares ONLY the two regular (no refinement) methods:
  1. NumPy:  Regular + NumPy exact solver
  2. QAOA:   Regular + QAOA solver

GD/Hybrid methods are EXCLUDED from this analysis.

Generates:
  - 2 heatmaps per metric (NumPy and QAOA)
  - 1 difference heatmap per metric
  - Comparison line plots across ALL budgets and ALL q values
  - Statistical comparison tables
  - Winner analysis

Usage:
    python analyze_results_numpy_vs_qaoa.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats
from qfolio.analysis.load_results_helper import (
    load_all_results,
    calculate_consistency_score
)

# Configuration
RESULTS_DIR = "results"
FIGURES_DIR = "figures_numpy_vs_qaoa"
TABLES_DIR = "tables_numpy_vs_qaoa"
DPI = 300
FIGSIZE_HEATMAP = (12, 8)
FIGSIZE_LINE = (14, 6)
CMAP_SEQUENTIAL = 'viridis'
CMAP_DIVERGING = 'RdYlGn'

# Create output directories
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# Method configuration - ONLY REGULAR METHODS
METHOD_LABELS = {
    ('regular', 'classic'): 'NumPy',
    ('regular', 'QAOA'): 'QAOA'
}

METHOD_COLORS = {
    'NumPy': '#1f77b4',  # Blue
    'QAOA': '#ff7f0e'    # Orange
}

METHOD_MARKERS = {
    'NumPy': 'o',
    'QAOA': 's'
}


def filter_regular_only(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to keep only regular methods (no hybrid/GD)."""
    return df[df['method'] == 'regular'].copy()


def categorize_method(row):
    """Create method label from method + solver_type."""
    key = (row['method'], row['solver_type'])
    return METHOD_LABELS.get(key, 'Unknown')


def generate_heatmap(pivot: pd.DataFrame, title: str, filename: str,
                     fmt: str = '.2f', cmap: str = CMAP_SEQUENTIAL,
                     center: float = None):
    """Generate a single heatmap."""
    fig, ax = plt.subplots(figsize=FIGSIZE_HEATMAP)

    sns.heatmap(
        pivot,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        center=center,
        cbar_kws={'label': title.split('(')[0].strip()},
        linewidths=0.5,
        linecolor='gray',
        ax=ax
    )

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Budget ($)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Risk Aversion (q)', fontsize=13, fontweight='bold')

    # Format budget labels
    budget_labels = [f"${int(b/1000)}k" if b < 1000000 else f"${int(b/1000000)}M"
                     for b in pivot.columns]
    ax.set_xticklabels(budget_labels, rotation=45, ha='right', fontsize=11)

    # Format q labels
    q_labels = [f"{q:.0e}" for q in pivot.index]
    ax.set_yticklabels(q_labels, rotation=0, fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=DPI, bbox_inches='tight')
    plt.close()

    print(f"[OK] Saved: {filename}")


def generate_all_heatmaps(df: pd.DataFrame):
    """Generate heatmaps for NumPy and QAOA."""
    print("\n" + "="*90)
    print("GENERATING HEATMAPS (NumPy vs QAOA)")
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

    df['consistency_score'] = calculate_consistency_score(df)
    df['method_label'] = df.apply(categorize_method, axis=1)

    for metric, (metric_name, fmt) in metrics.items():
        print(f"\n{metric_name}:")

        # NumPy heatmap
        numpy_df = df[df['method_label'] == 'NumPy']
        if len(numpy_df) > 0:
            pivot_numpy = numpy_df.pivot_table(
                values=metric,
                index='q',
                columns='budget',
                aggfunc='first'
            )
            generate_heatmap(
                pivot_numpy,
                f"{metric_name} - NumPy (Regular)",
                f"heatmap_{metric}_NumPy.png",
                fmt=fmt
            )

        # QAOA heatmap
        qaoa_df = df[df['method_label'] == 'QAOA']
        if len(qaoa_df) > 0:
            pivot_qaoa = qaoa_df.pivot_table(
                values=metric,
                index='q',
                columns='budget',
                aggfunc='first'
            )
            generate_heatmap(
                pivot_qaoa,
                f"{metric_name} - QAOA (Regular)",
                f"heatmap_{metric}_QAOA.png",
                fmt=fmt
            )

            # Difference heatmap (QAOA - NumPy)
            if len(numpy_df) > 0:
                pivot_diff = pivot_qaoa - pivot_numpy
                generate_heatmap(
                    pivot_diff,
                    f"{metric_name} - Difference (QAOA - NumPy)",
                    f"heatmap_{metric}_difference.png",
                    fmt=fmt,
                    cmap=CMAP_DIVERGING,
                    center=0
                )


def generate_comparison_line_plots(df: pd.DataFrame):
    """Generate line plots comparing NumPy vs QAOA across ALL budgets and q values."""
    print("\n" + "="*90)
    print("GENERATING COMPARISON LINE PLOTS (ALL BUDGETS & Q VALUES)")
    print("="*90)

    df['method_label'] = df.apply(categorize_method, axis=1)

    metrics = {
        'sharpe_ratio': 'Sharpe Ratio',
        'calmar_ratio': 'Calmar Ratio',
        'cagr_pct': 'CAGR (%)',
        'max_drawdown_pct': 'Max Drawdown (%)',
        'consistency_score': 'Consistency Score',
        'max_single_asset_weight_pct': 'Max Concentration (%)'
    }

    # Get ALL unique budgets and q values from the data
    all_budgets = sorted(df['budget'].unique())
    all_q_values = sorted(df['q'].unique())

    print(f"\nBudgets to plot: {all_budgets}")
    print(f"Q values to plot: {all_q_values}")

    # 1. Metric vs q (for ALL budgets)
    for metric, metric_name in metrics.items():
        for budget in all_budgets:
            fig, ax = plt.subplots(figsize=FIGSIZE_LINE)

            for method_label in ['NumPy', 'QAOA']:
                method_data = df[(df['method_label'] == method_label) & (df['budget'] == budget)]

                if len(method_data) == 0:
                    continue

                method_data_sorted = method_data.sort_values('q')
                ax.plot(method_data_sorted['q'], method_data_sorted[metric],
                       marker=METHOD_MARKERS[method_label],
                       color=METHOD_COLORS[method_label],
                       label=method_label,
                       linewidth=2.5,
                       markersize=9,
                       alpha=0.8)

            ax.set_xscale('log')
            ax.set_xlabel('Risk Aversion (q)', fontsize=13, fontweight='bold')
            ax.set_ylabel(metric_name, fontsize=13, fontweight='bold')

            budget_label = f"${int(budget/1000)}k" if budget < 1000000 else f"${int(budget/1000000)}M"
            ax.set_title(f'{metric_name} vs q (Budget = {budget_label})',
                        fontsize=15, fontweight='bold')
            ax.legend(loc='best', frameon=True, shadow=True, fontsize=12)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            filename = f"lineplot_{metric}_vs_q_budget{int(budget)}.png"
            plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=DPI, bbox_inches='tight')
            plt.close()

            print(f"[OK] Saved: {filename}")

    # 2. Metric vs Budget (for ALL q values)
    for metric, metric_name in metrics.items():
        for q in all_q_values:
            fig, ax = plt.subplots(figsize=FIGSIZE_LINE)

            for method_label in ['NumPy', 'QAOA']:
                method_data = df[(df['method_label'] == method_label) & (df['q'] == q)]

                if len(method_data) == 0:
                    continue

                method_data_sorted = method_data.sort_values('budget')
                ax.plot(method_data_sorted['budget'], method_data_sorted[metric],
                       marker=METHOD_MARKERS[method_label],
                       color=METHOD_COLORS[method_label],
                       label=method_label,
                       linewidth=2.5,
                       markersize=9,
                       alpha=0.8)

            ax.set_xscale('log')
            ax.set_xlabel('Budget ($)', fontsize=13, fontweight='bold')
            ax.set_ylabel(metric_name, fontsize=13, fontweight='bold')
            ax.set_title(f'{metric_name} vs Budget (q = {q:.0e})',
                        fontsize=15, fontweight='bold')
            ax.legend(loc='best', frameon=True, shadow=True, fontsize=12)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            filename = f"lineplot_{metric}_vs_budget_q{q:.0e}.png"
            plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=DPI, bbox_inches='tight')
            plt.close()

            print(f"[OK] Saved: {filename}")


def generate_bar_charts(df: pd.DataFrame):
    """Generate bar charts comparing average performance."""
    print("\n" + "="*90)
    print("GENERATING BAR CHARTS")
    print("="*90)

    df['method_label'] = df.apply(categorize_method, axis=1)

    metrics = {
        'sharpe_ratio': 'Sharpe Ratio',
        'calmar_ratio': 'Calmar Ratio',
        'cagr_pct': 'CAGR (%)',
        'max_drawdown_pct': 'Max Drawdown (%)',
        'consistency_score': 'Consistency Score',
        'max_single_asset_weight_pct': 'Max Concentration (%)'
    }

    for metric, metric_name in metrics.items():
        fig, ax = plt.subplots(figsize=(10, 6))

        # Calculate mean and std for each method
        method_stats = df.groupby('method_label')[metric].agg(['mean', 'std']).reset_index()

        # Sort by mean
        if 'drawdown' in metric.lower() or 'concentration' in metric.lower():
            method_stats = method_stats.sort_values('mean', ascending=True)
        else:
            method_stats = method_stats.sort_values('mean', ascending=False)

        # Create bar chart
        colors = [METHOD_COLORS[label] for label in method_stats['method_label']]

        bars = ax.bar(range(len(method_stats)), method_stats['mean'],
                     yerr=method_stats['std'],
                     color=colors,
                     alpha=0.7,
                     capsize=8,
                     edgecolor='black',
                     linewidth=2)

        ax.set_xticks(range(len(method_stats)))
        ax.set_xticklabels(method_stats['method_label'], fontsize=14, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=13, fontweight='bold')
        ax.set_title(f'Average {metric_name} (±1 std)\nNumPy vs QAOA',
                    fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, mean_val, std_val in zip(bars, method_stats['mean'], method_stats['std']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std_val,
                   f'{mean_val:.2f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=12)

        plt.tight_layout()
        filename = f"barchart_{metric}_comparison.png"
        plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=DPI, bbox_inches='tight')
        plt.close()

        print(f"[OK] Saved: {filename}")


def perform_statistical_tests(df: pd.DataFrame):
    """Perform paired t-tests comparing NumPy vs QAOA."""
    print("\n" + "="*90)
    print("PERFORMING STATISTICAL TESTS (NumPy vs QAOA)")
    print("="*90)

    df['method_label'] = df.apply(categorize_method, axis=1)

    metrics = {
        'sharpe_ratio': 'Sharpe Ratio',
        'calmar_ratio': 'Calmar Ratio',
        'cagr_pct': 'CAGR (%)',
        'consistency_score': 'Consistency Score'
    }

    results = []

    for metric, metric_name in metrics.items():
        print(f"\n{metric_name}:")

        # Get matched pairs (same q and budget)
        numpy_df = df[df['method_label'] == 'NumPy'].set_index(['q', 'budget'])
        qaoa_df = df[df['method_label'] == 'QAOA'].set_index(['q', 'budget'])

        # Find common indices
        common_idx = numpy_df.index.intersection(qaoa_df.index)

        if len(common_idx) == 0:
            print("  [WARNING] No matched pairs for t-test")
            continue

        numpy_values = numpy_df.loc[common_idx, metric].values
        qaoa_values = qaoa_df.loc[common_idx, metric].values

        # Paired t-test
        t_stat, p_value = scipy_stats.ttest_rel(qaoa_values, numpy_values)

        # Calculate means and effect size
        numpy_mean = numpy_values.mean()
        qaoa_mean = qaoa_values.mean()
        diff = qaoa_mean - numpy_mean
        pct_diff = 100 * diff / numpy_mean if numpy_mean != 0 else 0

        # Cohen's d effect size
        diff_values = qaoa_values - numpy_values
        cohens_d = diff_values.mean() / diff_values.std() if diff_values.std() > 0 else 0

        sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'

        print(f"  NumPy mean: {numpy_mean:.3f}")
        print(f"  QAOA mean:  {qaoa_mean:.3f}")
        print(f"  Difference: {diff:+.3f} ({pct_diff:+.1f}%)")
        print(f"  t-test: t={t_stat:.2f}, p={p_value:.4f} {sig}")
        print(f"  Effect size (Cohen's d): {cohens_d:.3f}")

        results.append({
            'metric': metric_name,
            'numpy_mean': numpy_mean,
            'qaoa_mean': qaoa_mean,
            'difference': diff,
            'pct_difference': pct_diff,
            't_stat': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'n_pairs': len(common_idx)
        })

    # Save results table
    results_df = pd.DataFrame(results)

    latex_table = """\\begin{table}[htbp]
\\centering
\\caption{Statistical Comparison: NumPy vs QAOA (Paired t-tests)}
\\label{tab:numpy_qaoa_stats}
\\begin{tabular}{lcccccc}
\\toprule
Metric & NumPy & QAOA & Diff & \\% Diff & t-stat & p-value \\\\
\\midrule
"""

    for _, row in results_df.iterrows():
        sig = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row['p_value'] < 0.05 else ''
        latex_table += f"{row['metric']} & {row['numpy_mean']:.2f} & {row['qaoa_mean']:.2f} & "
        latex_table += f"{row['difference']:+.2f} & {row['pct_difference']:+.1f}\\% & "
        latex_table += f"{row['t_stat']:.2f} & {row['p_value']:.4f}{sig} \\\\\n"

    latex_table += """\\bottomrule
\\multicolumn{7}{l}{\\footnotesize *** p<0.001, ** p<0.01, * p<0.05} \\\\
\\end{tabular}
\\end{table}
"""

    with open(os.path.join(TABLES_DIR, 'table_statistical_comparison.tex'), 'w') as f:
        f.write(latex_table)

    print(f"\n[OK] Saved: table_statistical_comparison.tex")


def generate_text_summary(df: pd.DataFrame):
    """Generate comprehensive text summary."""
    print("\n" + "="*90)
    print("GENERATING TEXT SUMMARY")
    print("="*90)

    df['method_label'] = df.apply(categorize_method, axis=1)

    summary = []
    summary.append("="*90)
    summary.append("NUMPY vs QAOA COMPARISON (REGULAR METHODS ONLY - NO GD)")
    summary.append("="*90)
    summary.append("")
    summary.append("METHODS COMPARED:")
    summary.append("-"*90)
    summary.append("  1. NumPy: Regular method + NumPy exact solver (no refinement)")
    summary.append("  2. QAOA:  Regular method + QAOA solver (no refinement)")
    summary.append("")
    summary.append("NOTE: All hybrid/GD methods are EXCLUDED from this analysis.")
    summary.append("")

    # Dataset overview
    summary.append("DATASET OVERVIEW")
    summary.append("-"*90)
    summary.append(f"Total backtests: {len(df)}")
    for method_label in ['NumPy', 'QAOA']:
        count = len(df[df['method_label'] == method_label])
        summary.append(f"  - {method_label}: {count}")
    summary.append(f"Grid search dimensions:")
    summary.append(f"  - q values: {sorted(df['q'].unique())}")
    summary.append(f"  - Budgets: {sorted(df['budget'].unique())}")
    summary.append("")

    # Average performance
    summary.append("AVERAGE PERFORMANCE")
    summary.append("-"*90)

    metrics = ['sharpe_ratio', 'calmar_ratio', 'cagr_pct', 'max_drawdown_pct',
               'consistency_score', 'max_single_asset_weight_pct']
    metric_names = ['Sharpe Ratio', 'Calmar Ratio', 'CAGR (%)', 'Max Drawdown (%)',
                   'Consistency Score', 'Max Concentration (%)']

    for metric, metric_name in zip(metrics, metric_names):
        summary.append(f"\n{metric_name}:")
        for method_label in ['NumPy', 'QAOA']:
            mean_val = df[df['method_label'] == method_label][metric].mean()
            std_val = df[df['method_label'] == method_label][metric].std()
            summary.append(f"  {method_label:8s}: {mean_val:.2f} ± {std_val:.2f}")

    summary.append("")

    # Winner counts
    summary.append("WINNER ANALYSIS (Configuration-by-Configuration)")
    summary.append("-"*90)

    # Merge NumPy and QAOA on (q, budget)
    numpy_df = df[df['method_label'] == 'NumPy'].set_index(['q', 'budget'])
    qaoa_df = df[df['method_label'] == 'QAOA'].set_index(['q', 'budget'])
    common_idx = numpy_df.index.intersection(qaoa_df.index)

    if len(common_idx) > 0:
        for metric, metric_name in zip(metrics, metric_names):
            numpy_vals = numpy_df.loc[common_idx, metric]
            qaoa_vals = qaoa_df.loc[common_idx, metric]

            if 'drawdown' in metric.lower() or 'concentration' in metric.lower():
                # Lower is better
                numpy_wins = (numpy_vals < qaoa_vals).sum()
                qaoa_wins = (qaoa_vals < numpy_vals).sum()
            else:
                # Higher is better
                numpy_wins = (numpy_vals > qaoa_vals).sum()
                qaoa_wins = (qaoa_vals > numpy_vals).sum()

            ties = len(common_idx) - numpy_wins - qaoa_wins

            summary.append(f"\n{metric_name}:")
            summary.append(f"  NumPy wins: {numpy_wins}/{len(common_idx)} ({100*numpy_wins/len(common_idx):.1f}%)")
            summary.append(f"  QAOA wins:  {qaoa_wins}/{len(common_idx)} ({100*qaoa_wins/len(common_idx):.1f}%)")
            if ties > 0:
                summary.append(f"  Ties:       {ties}/{len(common_idx)} ({100*ties/len(common_idx):.1f}%)")

    summary.append("")
    summary.append("="*90)

    # Write to file
    summary_text = "\n".join(summary)
    with open('analysis_summary_numpy_vs_qaoa.txt', 'w') as f:
        f.write(summary_text)

    print(summary_text)
    print("\n[OK] Saved: analysis_summary_numpy_vs_qaoa.txt")


def main():
    """Run complete NumPy vs QAOA analysis."""
    print("\n" + "="*90)
    print("NUMPY vs QAOA COMPARISON (NO GD)")
    print("="*90)

    # Load results
    print(f"\nLoading results from {RESULTS_DIR}/...")
    df = load_all_results(RESULTS_DIR)

    # Filter to regular methods only
    print(f"\nFiltering: Keeping ONLY regular methods (no hybrid/GD)...")
    df_before = len(df)
    df = filter_regular_only(df)
    df_after = len(df)
    print(f"  Removed {df_before - df_after} hybrid/GD results")
    print(f"  Remaining: {df_after} regular backtests")

    # Filter out q>=0.2 outlier
    print(f"\nFiltering: Removing q>=0.2 outlier...")
    df_before = len(df)
    df = df[df['q'] < 0.2].copy()
    df_after = len(df)
    print(f"  Removed {df_before - df_after} rows with q>=0.2")
    print(f"  Remaining: {df_after} backtests")

    # Check data coverage
    print("\nData coverage:")
    for method_label in ['NumPy', 'QAOA']:
        df['method_label'] = df.apply(categorize_method, axis=1)
        count = len(df[df['method_label'] == method_label])
        print(f"  {method_label:8s}: {count} backtests")

    if len(df) == 0:
        print("\n[ERROR] No data remaining after filtering!")
        return

    # Generate all outputs
    generate_all_heatmaps(df)
    generate_comparison_line_plots(df)
    generate_bar_charts(df)
    perform_statistical_tests(df)
    generate_text_summary(df)

    print("\n" + "="*90)
    print("NUMPY vs QAOA ANALYSIS COMPLETE!")
    print("="*90)
    print(f"\nOutputs:")
    print(f"  - Figures:  {FIGURES_DIR}/")
    print(f"  - Tables:   {TABLES_DIR}/")
    print(f"  - Summary:  analysis_summary_numpy_vs_qaoa.txt")
    print("\n")


if __name__ == '__main__':
    main()
