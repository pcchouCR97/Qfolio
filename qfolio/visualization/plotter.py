import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns

def plot_dv_comparison(port1, port2, initial_budget, exchange):

    """
    plot daily value comparison

    port1: optimized portfolio (Series)
    port2: benchmark target (DataFrame or Series)
    initial_budget: initial investment amount
    exchange: exchange name for title
    """
    # Extract SPY series if port2 is a DataFrame
    spy_series = port2['SPY'] if isinstance(port2, pd.DataFrame) else port2

    plt.figure(figsize=(14, 7))
    plt.plot(port1, label="Optimized Portfolio", linewidth=2.5, color='black')
    plt.plot(spy_series, label="SPY Benchmark", linewidth=2.5, linestyle='--', color='blue')

    # Add initial investment reference line
    plt.axhline(y=initial_budget, color='gray', linestyle=':', alpha=0.5, linewidth=1, label=f'Initial: ${initial_budget}')

    # Shade regions where optimized beats SPY
    opt_beats_spy = port1 > spy_series
    if opt_beats_spy.any():
        plt.fill_between(port1.index, port1, spy_series,
                        where=opt_beats_spy, alpha=0.2, color='green',
                        label='Optimized Outperforms', interpolate=True)
        plt.fill_between(port1.index, port1, spy_series,
                        where=~opt_beats_spy, alpha=0.2, color='red',
                        label='SPY Outperforms', interpolate=True)

    plt.title(f"Portfolio Value Comparison: Optimized vs SPY ({exchange})",
            fontsize=14, fontweight='bold')
    plt.ylabel("Portfolio Value ($)", fontsize=12)
    plt.xlabel("Date", fontsize=12)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return None

def plot_roi_comp(port_roi, bm_roi, all_rebalance_events, exchange):
    # Extract SPY series if bm_roi is a DataFrame
    spy_roi = bm_roi['SPY'] if isinstance(bm_roi, pd.DataFrame) else bm_roi

    plt.figure(figsize=(12, 6))
    plt.plot(port_roi, label="Optimized Portfolio ROI", linewidth=2, color='black')
    plt.plot(spy_roi, label="100% SPY Benchmark ROI", linewidth=2, linestyle='--', color='blue')

    # Highlight ALL rebalance events
    if all_rebalance_events:
        for i, date in enumerate(all_rebalance_events):
            plt.axvline(x=date, color='red', linestyle='--', linewidth=1, label='Rebalance Date' if i == 0 else "")

    plt.title(f"Portfolio ROI vs. 100% SPY Benchmark ({exchange})")
    plt.xlabel("Date")
    plt.ylabel("Return on Investment (%)")
    plt.axhline(0, color='grey', linestyle='--') # Add a zero line for reference
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return None

def plot_port_val(port_values, exchange):
    # --- Original Portfolio Value Chart (Keep for reference) ---
    plt.figure(figsize=(12, 6))
    plt.plot(port_values, label="Optimized Portfolio Value", linewidth=2, color='black')
    plt.title(f"Portfolio Value Over Time ({exchange})")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return None


def plot_composition_n_turnover(composition_history, turnover_history, sharpe_n):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Number of assets over time
    comp_df = pd.DataFrame(composition_history)
    comp_df['date'] = pd.to_datetime(comp_df['date'])
    comp_df = comp_df.set_index('date')

    ax1.plot(comp_df.index, comp_df['num_assets'], linewidth=2, color='darkblue', marker='o', markersize=6)
    ax1.set_title("Portfolio Size Over Time", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Number of Assets", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=sharpe_n, color='red', linestyle='--', alpha=0.5, linewidth=2, label=f'Target: {sharpe_n}')
    ax1.legend()
    ax1.set_ylim(bottom=0)

    # Turnover over time
    turnover_df = pd.DataFrame(turnover_history)
    turnover_df['date'] = pd.to_datetime(turnover_df['date'])
    turnover_df = turnover_df.set_index('date')

    ax2.bar(turnover_df.index, turnover_df['turnover'], width=20, color='orange', alpha=0.7, edgecolor='black')
    ax2.set_title("Portfolio Turnover at Each Rebalance", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Date", fontsize=12)
    ax2.set_ylabel("Turnover (%)", fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=50, color='red', linestyle='--', alpha=0.3, linewidth=2, label='50% threshold')
    ax2.legend()

    plt.tight_layout()
    plt.show()
    return

def plot_sector_exposure(sector_exposure_history):

    sector_df = pd.DataFrame(sector_exposure_history)
    sector_df['date'] = pd.to_datetime(sector_df['date'])
    sector_df = sector_df.set_index('date')
    sector_df = sector_df.fillna(0)

    # Sort sectors by total allocation (most popular on top)
    sector_totals = sector_df.sum().sort_values(ascending=False)
    sector_df_sorted = sector_df[sector_totals.index]

    plt.figure(figsize=(14, 8))
    sector_df_sorted.plot.area(stacked=True, alpha=0.7, figsize=(14, 8),
                                colormap='tab20', linewidth=0, ax=plt.gca())
    plt.title("Sector Allocation Over Time (Concentrated Strategy)",
              fontsize=14, fontweight='bold')
    plt.ylabel("Portfolio Weight (%)", fontsize=12)
    plt.xlabel("Rebalance Date", fontsize=12)
    plt.legend(title="Sector", bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=9)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

    return None

def plot_se_history(sector_exposure_history):

    """
    plot sector exposure history
    """

    sector_df = pd.DataFrame(sector_exposure_history)
    sector_df['date'] = pd.to_datetime(sector_df['date'])
    sector_df = sector_df.set_index('date')
    sector_df = sector_df.fillna(0)

    # Sort sectors by total allocation
    sector_totals = sector_df.sum().sort_values(ascending=False)
    sector_df_sorted = sector_df[sector_totals.index]

    plt.figure(figsize=(16, 8))
    sns.heatmap(sector_df_sorted.T, cmap='YlOrRd', cbar_kws={'label': 'Allocation (%)'},
                linewidths=0.5, linecolor='lightgray', vmin=0, vmax=100)
    plt.title("Sector Exposure Heatmap (White = Not Held)", fontsize=14, fontweight='bold')
    plt.xlabel("Rebalance Date", fontsize=12)
    plt.ylabel("Sector", fontsize=12)

    # Format x-axis to show fewer dates
    n_dates = len(sector_df_sorted)
    step = max(1, n_dates // 10)  # Show ~10 dates
    plt.xticks(range(0, n_dates, step),
               [sector_df_sorted.index[i].strftime('%Y-%m-%d') for i in range(0, n_dates, step)],
               rotation=45)

    plt.tight_layout()
    plt.show()

    return None

def plot_conecntration(concentration_history):

    conc_df = pd.DataFrame(concentration_history)
    conc_df['date'] = pd.to_datetime(conc_df['date'])
    conc_df = conc_df.set_index('date')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # HHI over time
    ax1.plot(conc_df.index, conc_df['HHI'], linewidth=2, color='darkred', marker='o', markersize=5)
    ax1.set_title("Sector Concentration (HHI) Over Time", fontsize=14, fontweight='bold')
    ax1.set_ylabel("HHI", fontsize=12)
    ax1.axhline(y=25, color='red', linestyle='--', alpha=0.5, linewidth=2, label='High Concentration (>25)')
    ax1.axhline(y=15, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='Moderate (15-25)')
    ax1.fill_between(conc_df.index, 0, 15, alpha=0.1, color='green', label='Diversified (<15)')
    ax1.fill_between(conc_df.index, 15, 25, alpha=0.1, color='orange')
    ax1.fill_between(conc_df.index, 25, 100, alpha=0.1, color='red')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)

    # Number of sectors held over time
    ax2.plot(conc_df.index, conc_df['num_sectors'], linewidth=2, color='darkgreen', marker='s', markersize=5)
    ax2.set_title("Number of Sectors Held Over Time", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Rebalance Date", fontsize=12)
    ax2.set_ylabel("Number of Sectors", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 12)
    ax2.axhline(y=11, color='gray', linestyle=':', alpha=0.5, linewidth=2, label='Max possible (11 sectors)')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    return None