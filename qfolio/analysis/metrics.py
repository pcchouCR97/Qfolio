import numpy as np
import quantstats as qs
from qfolio.analysis.comparison import *

def calculate_advanced_metrics(opt_daily_returns, spy_daily_returns, opt_drawdown):
    """
    Calculate all advanced QuantStats metrics for export.

    Returns dict with all advanced metrics for use with save_backtest_results().
    """
    # Greeks
    greeks = qs.stats.greeks(opt_daily_returns, spy_daily_returns)
    opt_alpha = greeks['alpha'] * 252 * 100  # Annualized in %
    opt_beta = greeks['beta']

    # Risk-adjusted returns
    opt_r2 = qs.stats.r_squared(opt_daily_returns, spy_daily_returns)
    opt_treynor = (opt_daily_returns.mean() * 252) / opt_beta if opt_beta != 0 else 0
    opt_sortino = qs.stats.sortino(opt_daily_returns)
    opt_calmar = qs.stats.calmar(opt_daily_returns)
    opt_cagr = qs.stats.cagr(opt_daily_returns) * 100
    opt_information_ratio = qs.stats.information_ratio(opt_daily_returns, spy_daily_returns)

    # Performance ratios
    opt_omega = qs.stats.omega(opt_daily_returns)
    opt_gain_to_pain = qs.stats.gain_to_pain_ratio(opt_daily_returns)
    opt_payoff = qs.stats.payoff_ratio(opt_daily_returns)
    opt_profit_factor = qs.stats.profit_factor(opt_daily_returns)
    opt_recovery_factor = qs.stats.recovery_factor(opt_daily_returns)
    opt_tail_ratio = qs.stats.tail_ratio(opt_daily_returns)

    # Distribution
    opt_skew = qs.stats.skew(opt_daily_returns)
    opt_kurtosis = qs.stats.kurtosis(opt_daily_returns)

    # Market capture
    upside_capture, downside_capture, capture_ratio = up_downside_captrue_ratio(opt_daily_returns, spy_daily_returns)

    # Drawdown duration
    max_drawdown_days, avg_drawdown_days, spy_max_drawdown_days, spy_avg_drawdown_days = drawdown_duration(opt_drawdown, spy_daily_returns)

    return {
        'alpha_pct': float(opt_alpha),
        'beta': float(opt_beta),
        'r_squared': float(opt_r2),
        'treynor_ratio': float(opt_treynor),
        'sortino_ratio': float(opt_sortino),
        'calmar_ratio': float(opt_calmar),
        'cagr_pct': float(opt_cagr),
        'information_ratio': float(opt_information_ratio),
        'omega_ratio': float(opt_omega),
        'profit_factor': float(opt_profit_factor),
        'gain_to_pain_ratio': float(opt_gain_to_pain),
        'recovery_factor': float(opt_recovery_factor),
        'payoff_ratio': float(opt_payoff),
        'tail_ratio': float(opt_tail_ratio),
        'skewness': float(opt_skew),
        'kurtosis': float(opt_kurtosis),
        'upside_capture_pct': float(upside_capture),
        'downside_capture_pct': float(abs(downside_capture)),
        'capture_ratio': float(capture_ratio),
        'max_drawdown_duration_days': int(max_drawdown_days),
        'avg_drawdown_duration_days': float(avg_drawdown_days)
    }

def calculate_performance_stats(portfolio_history, port_values, port_roi):
    
    stats = {}

    # Days in market 
    #days_in_market = sum(1 for record in portfolio_history if len(record['allocation']) > 0)
    #total_days = len(portfolio_history)
    # Annualized metrics
    trading_days_per_year = 252
    total_days = len(portfolio_history)
    
    # --- Calculate Performance Statistics ---
    # Optimized Portfolio Statistics
    opt_daily_returns = port_values.pct_change().dropna()
    opt_total_return = port_roi.iloc[-1]
    opt_volatility = opt_daily_returns.std() * np.sqrt(252) * 100
    opt_sharpe = (opt_daily_returns.mean() / opt_daily_returns.std()) * np.sqrt(252) if opt_daily_returns.std() > 0 else 0

    # Annualized return
    years = total_days / trading_days_per_year
    annualized_return = ((1 + opt_total_return/100) ** (1/years) - 1) * 100 if years > 0 else 0


    # Calculate max drawdown for optimized portfolio
    opt_cumulative = (1 + opt_daily_returns).cumprod()
    opt_running_max = opt_cumulative.cummax()
    opt_drawdown = (opt_cumulative - opt_running_max) / opt_running_max
    opt_max_drawdown = opt_drawdown.min() * 100

    # Win rate for optimized portfolio
    opt_win_rate = (opt_daily_returns > 0).sum() / len(opt_daily_returns) * 100 if len(opt_daily_returns) > 0 else 0

    stats = {
            'Daily Return (%)': opt_daily_returns,
            'Total Return (%)': opt_total_return,
            'Annualized Return (%)': annualized_return,
            'Volatility (%)': opt_volatility,
            'Sharpe Ratio': opt_sharpe,
            'Drawdown (%)': opt_drawdown,
            'Max Drawdown (%)': opt_max_drawdown,
            'Win Rate (%)': opt_win_rate,
            'Total Days': total_days
            }
    
    return stats

def key_metrics(portfolio_history, port_values, port_roi, bm_values, bm_roi, bm, advance = False):
    
    # --- Calculate ALL Key Metrics ---
    #========= optimized portfolio metrics =========#
    # Optimized Portfolio Statistics
    
    port_stats = calculate_performance_stats(portfolio_history, 
                                         port_values,
                                        port_roi)
    
    opt_daily_returns = port_stats['Daily Return (%)']
    opt_total_return = port_stats['Total Return (%)']
    opt_volatility = port_stats['Volatility (%)']
    opt_sharpe = port_stats['Sharpe Ratio']
    opt_drawdown = port_stats['Drawdown (%)']
    opt_max_drawdown = port_stats['Max Drawdown (%)']
    opt_win_rate = port_stats['Win Rate (%)']
    total_days = len(portfolio_history)

    # SPY Statistics
    spy_stats = bm.get_statistics(bm_values, bm_roi)['SPY']
    spy_daily_returns = bm_values['SPY'].pct_change().dropna()
    spy_total_return = spy_stats['Total Return (%)']
    spy_volatility = spy_stats['Volatility (%)']
    spy_sharpe = spy_stats['Sharpe Ratio']
    spy_max_drawdown = spy_stats['Max Drawdown (%)']
    spy_win_rate = spy_stats['Win Rate (%)']

    upside_capture, downside_capture, capture_ratio = up_downside_captrue_ratio(opt_daily_returns, spy_daily_returns)
    max_drawdown_days, avg_drawdown_days, spy_max_drawdown_days, spy_avg_drawdown_days = drawdown_duration(opt_drawdown, spy_daily_returns)

    # 1. Alpha & Beta (CRITICAL!)
    greeks = qs.stats.greeks(opt_daily_returns, spy_daily_returns)
    opt_alpha = greeks['alpha'] * 252 * 100  # Annualized in %
    opt_beta = greeks['beta']

    # 2. R-Squared
    opt_r2 = qs.stats.r_squared(opt_daily_returns, spy_daily_returns)

    # 3. Treynor Ratio (manual calculation)
    opt_treynor = (opt_daily_returns.mean() * 252) / opt_beta if opt_beta != 0 else 0

    # 4. Already in previous list
    opt_sortino = qs.stats.sortino(opt_daily_returns)
    opt_calmar = qs.stats.calmar(opt_daily_returns)
    opt_cagr = qs.stats.cagr(opt_daily_returns) * 100
    opt_information_ratio = qs.stats.information_ratio(opt_daily_returns, spy_daily_returns)

    # 5. Additional metrics
    opt_omega = qs.stats.omega(opt_daily_returns)
    opt_gain_to_pain = qs.stats.gain_to_pain_ratio(opt_daily_returns)
    opt_payoff = qs.stats.payoff_ratio(opt_daily_returns)
    opt_profit_factor = qs.stats.profit_factor(opt_daily_returns)
    opt_max_drawdown = qs.stats.max_drawdown(opt_daily_returns) * 100
    opt_recovery_factor = qs.stats.recovery_factor(opt_daily_returns)
    opt_tail_ratio = qs.stats.tail_ratio(opt_daily_returns)
    opt_skew = qs.stats.skew(opt_daily_returns)
    opt_kurtosis = qs.stats.kurtosis(opt_daily_returns)

    #========== SPY portfolio metrics =========#
    # 1. Alpha & Beta (CRITICAL!)
    spy_greeks = qs.stats.greeks(spy_daily_returns, spy_daily_returns)
    spy_alpha = spy_greeks['alpha'] * 252 * 100  # Annualized in %
    spy_beta = spy_greeks['beta']

    # 2. R-Squared
    spy_r2 = qs.stats.r_squared(spy_daily_returns, spy_daily_returns)

    # 3. Treynor Ratio (manual calculation)
    spy_treynor = (spy_daily_returns.mean() * 252) / opt_beta if opt_beta != 0 else 0

    # 4. Already in previous list
    spy_sortino = qs.stats.sortino(spy_daily_returns)
    spy_calmar = qs.stats.calmar(spy_daily_returns)
    spy_cagr = qs.stats.cagr(spy_daily_returns) * 100
    spy_information_ratio = qs.stats.information_ratio(spy_daily_returns, spy_daily_returns)

    # 5. Additional metrics
    spy_omega = qs.stats.omega(spy_daily_returns)
    spy_gain_to_pain = qs.stats.gain_to_pain_ratio(spy_daily_returns)
    spy_payoff = qs.stats.payoff_ratio(spy_daily_returns)
    spy_profit_factor = qs.stats.profit_factor(spy_daily_returns)
    spy_max_drawdown = qs.stats.max_drawdown(spy_daily_returns) * 100
    spy_recovery_factor = qs.stats.recovery_factor(spy_daily_returns)
    spy_tail_ratio = qs.stats.tail_ratio(spy_daily_returns)
    spy_skew = qs.stats.skew(spy_daily_returns)
    spy_kurtosis = qs.stats.kurtosis(spy_daily_returns)

    # --- Statistical Significance Tests ---
    from scipy import stats

    excess_returns = opt_daily_returns - spy_daily_returns
    t_stat, p_value = stats.ttest_1samp(excess_returns.dropna(), 0)

    # Interpretation
    if p_value < 0.01:
        significance = "*** (Highly Significant)"
    elif p_value < 0.05:
        significance = "** (Significant)"
    elif p_value < 0.1:
        significance = "* (Marginally Significant)"
    else:
        significance = "(Not Significant)"

    # --- Print Comprehensive Comparison ---
    print("\n" + "="*90)
    print(" " * 30 + "COMPREHENSIVE PERFORMANCE COMPARISON")
    print("="*90)
    print(f"\n{'Metric':<40} {'Optimized':<22} {'SPY Benchmark':<22}")
    print("-"*90)

    # === RETURNS ===
    print("\n" + "RETURNS & GROWTH".center(90, " "))
    print(f"{'Total Return (%)':<40} {opt_total_return:>20.2f}% {spy_total_return:>20.2f}%")
    print(f"{'CAGR (%)':<40} {opt_cagr:>20.2f}% {spy_cagr:>20.2f}%")
    print(f"{'Alpha (annualized %)':<40} {opt_alpha:>20.2f}% {spy_alpha:>20.2f}%")

    # === RISK ===
    print("\n" + "RISK METRICS".center(90, " "))
    print(f"{'Volatility (annual %)':<40} {opt_volatility:>20.2f}% {spy_volatility:>20.2f}%")
    print(f"{'Beta':<40} {opt_beta:>20.2f}  {spy_beta:>20.2f}")
    print(f"{'R-Squared':<40} {opt_r2:>20.2f}  {spy_r2:>20.2f}")
    print(f"{'Max Drawdown (%)':<40} {opt_max_drawdown:>20.2f}% {spy_max_drawdown:>20.2f}%")

    # === RISK-ADJUSTED RETURNS ===
    print("\n" + "RISK-ADJUSTED RETURNS".center(90, " "))
    print(f"{'Sharpe Ratio':<40} {opt_sharpe:>20.2f}  {spy_sharpe:>20.2f}")
    print(f"{'Sortino Ratio':<40} {opt_sortino:>20.2f}  {spy_sortino:>20.2f}")
    print(f"{'Calmar Ratio':<40} {opt_calmar:>20.2f}  {spy_calmar:>20.2f}")
    print(f"{'Treynor Ratio':<40} {opt_treynor:>20.2f}  {spy_treynor:>20.2f}")
    print(f"{'Information Ratio':<40} {opt_information_ratio:>20.2f}  {'N/A':>20}")

    # === PERFORMANCE RATIOS ===
    print("\n" + "PERFORMANCE RATIOS".center(90, " "))
    print(f"{'Omega Ratio':<40} {opt_omega:>20.2f}  {spy_omega:>20.2f}")
    print(f"{'Profit Factor':<40} {opt_profit_factor:>20.2f}  {spy_profit_factor:>20.2f}")
    print(f"{'Gain-to-Pain Ratio':<40} {opt_gain_to_pain:>20.2f}  {spy_gain_to_pain:>20.2f}")
    print(f"{'Recovery Factor':<40} {opt_recovery_factor:>20.2f}  {spy_recovery_factor:>20.2f}")
    print(f"{'Payoff Ratio':<40} {opt_payoff:>20.2f}  {spy_payoff:>20.2f}")

    # === WIN/LOSS ===
    print("\n" + "WIN/LOSS STATISTICS".center(90, " "))
    print(f"{'Win Rate (%)':<40} {opt_win_rate:>20.2f}% {spy_win_rate:>20.2f}%")
    print(f"{'Tail Ratio':<40} {opt_tail_ratio:>20.2f}  {spy_tail_ratio:>20.2f}")

    # === DISTRIBUTION ===
    print("\n" + "RETURN DISTRIBUTION".center(90, " "))
    print(f"{'Skewness':<40} {opt_skew:>20.2f}  {spy_skew:>20.2f}")
    print(f"{'Kurtosis':<40} {opt_kurtosis:>20.2f}  {spy_kurtosis:>20.2f}")

    # === MARKET CAPTURE ===
    print("\n" + "MARKET CAPTURE".center(90, " "))
    print(f"{'Upside Capture (%)':<40} {upside_capture:>20.2f}% {'100.00%':>20}")
    print(f"{'Downside Capture (%)':<40} {abs(downside_capture):>20.2f}% {'100.00%':>20}")
    print(f"{'Capture Ratio':<40} {capture_ratio:>20.2f}  {'1.00':>20}")

    # === DRAWDOWN DURATION ===
    print("\n" + "DRAWDOWN ANALYSIS".center(90, " "))
    print(f"{'Max Drawdown Duration (days)':<40} {max_drawdown_days:>20.0f}  {spy_max_drawdown_days:>20.0f}")
    if avg_drawdown_days > 0:
        print(f"{'Avg Drawdown Duration (days)':<40} {avg_drawdown_days:>20.1f}  {spy_avg_drawdown_days:>20.1f}")

    if advance:
        """
        # === PORTFOLIO CHARACTERISTICS ===
        print("\n" + "PORTFOLIO CHARACTERISTICS".center(90, " "))
        if turnover_history:
            avg_turnover = sum(t['turnover'] for t in turnover_history) / len(turnover_history)
            print(f"{'Average Turnover per Rebalance (%)':<40} {avg_turnover:>20.2f}% {'0.00%':>20}")
        print(f"{'Number of Rebalances':<40} {len(actual_rebalance_dates):>20}  {len(actual_rebalance_dates):>20}")
        if composition_history:
            avg_assets = sum(c['num_assets'] for c in composition_history) / len(composition_history)
            print(f"{'Avg Assets per Period':<40} {avg_assets:>20.1f}  {'1.0':>20}")

        # === SECTOR DIVERSIFICATION ===
        if concentration_history:
            avg_hhi = sum(c['HHI'] for c in concentration_history) / len(concentration_history)
            avg_sectors = sum(c['num_sectors'] for c in concentration_history) / len(concentration_history)
            print("\n" + "SECTOR DIVERSIFICATION".center(90, " "))
            print(f"{'Average Sector HHI':<40} {avg_hhi:>20.1f}  {'N/A':>20}")
            hhi_interpretation = "(Highly Concentrated)" if avg_hhi > 25 else "(Moderate)" if avg_hhi > 15 else "(Diversified)"
            print(f"{'HHI Interpretation':<40} {hhi_interpretation:>42}")
            print(f"{'Avg Sectors Held per Period':<40} {avg_sectors:>20.1f}  {'11 (all)':>20}")
        """
    # === STATISTICAL TESTS ===
    print("\n" + "STATISTICAL SIGNIFICANCE".center(90, " "))
    print(f"{'Excess Return t-statistic':<40} {t_stat:>20.4f}  {'N/A':>20}")
    print(f"{'Excess Return p-value':<40} {p_value:>20.4f}  {'N/A':>20}")
    print(f"{'Significance Level':<40} {significance:>42}")

    print("-"*90)

    # Performance delta
    roi_diff = opt_total_return - spy_stats['Total Return (%)']
    sharpe_diff = opt_sharpe - spy_stats['Sharpe Ratio']
    print(f"\n{'Performance vs SPY:':<30}")
    print(f"{'  Return Difference':<30} {roi_diff:>18.2f}%  {'(Better)' if roi_diff > 0 else '(Worse)'}")
    print(f"{'  Sharpe Difference':<30} {sharpe_diff:>18.2f}   {'(Better)' if sharpe_diff > 0 else '(Worse)'}")
    print("="*70 + "\n")

    return None