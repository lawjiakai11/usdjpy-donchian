import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data import load_and_split
from src.backtest import performance_metrics
from src.validation import permute_ohlc, optimize_lookback, walk_forward

def setup_plot():
    plt.style.use('default')
    plt.rcParams.update({
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'axes.edgecolor': '#111111', 'axes.labelcolor': '#111111',
        'xtick.color': '#111111', 'ytick.color': '#111111',
        'grid.color': '#E5E5E5', 'text.color': '#111111'
    })

def plot_hist(distribution_data, real_sharpe, p_value, title, filename):
    setup_plot()
    plt.figure(figsize=(10, 6))
    plt.hist(distribution_data, bins=50, color='#E5E5E5', edgecolor='#111111')
    plt.axvline(real_sharpe, color='crimson', linewidth=2, label=f'Real Sharpe: {real_sharpe:.2f}')
    
    # Add P-Value text box to the plot
    plt.text(0.95, 0.95, f'P-Value: {p_value:.4f}', 
             horizontalalignment='right', verticalalignment='top', 
             transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='#111111', boxstyle='round,pad=0.5'))
    
    plt.title(title); plt.xlabel('Sharpe Ratio'); plt.ylabel('Frequency')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(filename); plt.close()

def plot_equity(real_equity_curve, permutation_equity_curves, title, filename):
    setup_plot()
    plt.figure(figsize=(10, 6))
    # Cloud of permutation paths
    for perm_curve in permutation_equity_curves: 
        plt.plot(perm_curve, color='grey', alpha=0.15, lw=0.5)
    # Real strategy curve
    plt.plot(real_equity_curve, color='crimson', linewidth=2, label='Real Strategy')
    plt.title(title); plt.xlabel('Date'); plt.ylabel('Log Equity')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(filename); plt.close()

def main():
    os.makedirs('reports/figures', exist_ok=True)
    with open('config.yaml') as f: 
        config = yaml.safe_load(f)

    in_sample_df, out_of_sample_df = load_and_split(config['data']['file_path'], config['oos_settings']['oos_months'])
    lookback_grid = config['optimization']['lookback_grid']
    fx_cost_params = config['fx_params']
    in_sample_runs, walk_forward_runs = config['permutation']['in_sample_runs'], config['permutation']['walk_forward_runs']
    np.random.seed(config['permutation']['seed'])

    ohlc_columns = ['open', 'high', 'low', 'close']
    
    print("Optimizing In-Sample...")
    best_lookback, best_sharpe, is_backtest_results = optimize_lookback(in_sample_df[ohlc_columns], lookback_grid, fx_cost_params, True)
    is_performance_metrics = performance_metrics(is_backtest_results)

    print(f"Running {in_sample_runs} In-Sample Permutations...")
    is_permuted_sharpes, is_permuted_equities_for_plot = [], []
    for i in range(in_sample_runs):
        permuted_df = permute_ohlc(in_sample_df[ohlc_columns])
        _, permuted_sharpe, permuted_results = optimize_lookback(permuted_df, lookback_grid, fx_cost_params, True)
        is_permuted_sharpes.append(permuted_sharpe)
        if i < 50: 
            is_permuted_equities_for_plot.append(permuted_results['cumulative_equity_curve'])
    
    # P-Value: What % of optimized noise beat the real optimized Sharpe?
    is_p_value = np.mean(np.array(is_permuted_sharpes) >= best_sharpe)

    print("Running Walk-Forward OOS...")
    oos_backtest_results = walk_forward(in_sample_df, out_of_sample_df, lookback_grid, fx_cost_params)
    oos_backtest_results["cumulative_equity_curve"] = oos_backtest_results["daily_strategy_return"].fillna(0).cumsum()
    oos_performance_metrics = performance_metrics(oos_backtest_results)
    oos_sharpe = oos_performance_metrics["Sharpe Ratio"]

    print(f"Running {walk_forward_runs} Walk-Forward Permutations...")
    full_dataset_df = pd.concat([in_sample_df, out_of_sample_df])
    oos_start_date = out_of_sample_df.index.min()
    
    wf_permuted_sharpes, wf_permuted_equities_for_plot = [], []
    for i in range(walk_forward_runs):
        permuted_full_df = permute_ohlc(full_dataset_df[ohlc_columns])
        permuted_full_df.index = full_dataset_df.index
        
        permuted_is_df = permuted_full_df[permuted_full_df.index < oos_start_date]
        permuted_oos_df = permuted_full_df[permuted_full_df.index >= oos_start_date]
        
        permuted_oos_results = walk_forward(permuted_is_df, permuted_oos_df, lookback_grid, fx_cost_params)
        wf_permuted_sharpes.append(performance_metrics(permuted_oos_results)["Sharpe Ratio"])
        
        if i < 50:
            permuted_oos_results["cumulative_equity_curve"] = permuted_oos_results["daily_strategy_return"].fillna(0).cumsum()
            wf_permuted_equities_for_plot.append(permuted_oos_results['cumulative_equity_curve'])
            
    wf_p_value = np.mean(np.array(wf_permuted_sharpes) >= oos_sharpe)

    print("Generating Plots...")
    plot_hist(is_permuted_sharpes, best_sharpe, is_p_value, "IS Sharpe Distribution", "reports/figures/is_sharpe_distribution.png")
    plot_equity(is_backtest_results['cumulative_equity_curve'], is_permuted_equities_for_plot, "IS Equity Curves", "reports/figures/is_equity_curves.png")
    plot_hist(wf_permuted_sharpes, oos_sharpe, wf_p_value, "OOS Sharpe Distribution", "reports/figures/oos_sharpe_distribution.png")
    plot_equity(oos_backtest_results['cumulative_equity_curve'], wf_permuted_equities_for_plot, "OOS Equity Curves", "reports/figures/oos_equity_curves.png")

    print("Generating Report...")
    report = f"""## Data & Split
- **Dataset:** `{config['data']['file_path']}`
- **In-Sample Period:** {in_sample_df.index.min().date()} to {in_sample_df.index.max().date()}
- **Out-of-Sample Period:** {out_of_sample_df.index.min().date()} to {out_of_sample_df.index.max().date()}

## FX Microstructure Parameters
- **Execution:** 5:00 PM NY Rollover
- **Transaction Cost:** {fx_cost_params['pip_cost']} pips round-turn
- **Carry Proxy:** {fx_cost_params['annual_carry'] * 100:.1f}% annualized interest rate differential

## Optimization Results
- **Optimized Lookback (IS):** {best_lookback} days
- **In-Sample Annualized Return:** {is_performance_metrics['Annualized Return'] * 100:.2f}%
- **In-Sample Sharpe Ratio:** {is_performance_metrics['Sharpe Ratio']:.4f}
- **In-Sample Sortino Ratio:** {is_performance_metrics['Sortino Ratio']:.4f}
- **In-Sample Profit Factor:** {is_performance_metrics['Profit Factor']:.4f}
- **In-Sample Max Drawdown:** {is_performance_metrics['Max Drawdown']:.4f}

## Out-of-Sample Performance
- **OOS Annualized Return:** {oos_performance_metrics['Annualized Return'] * 100:.2f}%
- **OOS Sharpe Ratio:** {oos_performance_metrics['Sharpe Ratio']:.4f}
- **OOS Sortino Ratio:** {oos_performance_metrics['Sortino Ratio']:.4f}
- **OOS Profit Factor:** {oos_performance_metrics['Profit Factor']:.4f}
- **OOS Max Drawdown:** {oos_performance_metrics['Max Drawdown']:.4f}

## Permutation Tests (Weekend-Preserving)
- **In-Sample P-Value:** {is_p_value:.4f} (Null: Sharpe from {in_sample_runs} permuted datasets >= real IS Sharpe)
- **Walk-Forward P-Value:** {wf_p_value:.4f} (Null: Sharpe from {walk_forward_runs} permuted WF runs >= real OOS Sharpe)

## Conclusion
The strategy exhibits {'statistically significant' if is_p_value < 0.05 and wf_p_value < 0.05 else 'insufficient'} evidence of persistent momentum after accounting for multiple comparisons, FX spreads, and swap carry.
"""
    with open('reports/research_summary.md', 'w') as f: 
        f.write(report)
    print("Done! Check reports/research_summary.md")

if __name__ == "__main__":
    main()