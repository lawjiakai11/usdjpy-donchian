import os
import yaml
import numpy as np
import pandas as pd
from src.data import load_and_split

def buy_and_hold_returns(df: pd.DataFrame, pip_cost: float = 0.5, point: float = 0.01, annual_carry: float = 0.02) -> pd.DataFrame:
    df = df.copy()
    
    # Always 100% Long from Day 1
    df["signal"] = 1.0
    
    # 1. Daily Asset Log Return
    df["daily_log_return"] = np.log(df["close"]).diff().shift(-1)
    df["daily_strategy_return"] = df["signal"] * df["daily_log_return"]

    # 2. Cost of Carry (Earn carry every single day)
    daily_carry = np.log(1 + annual_carry) / 252
    df["daily_strategy_return"] += df["signal"] * daily_carry

    # 3. Transaction Cost (Pay entry spread ONLY on the very first day of the period)
    entry_cost_log = np.log(1 - (pip_cost * point) / df["close"].iloc[0])
    df.loc[df.index[0], "daily_strategy_return"] -= entry_cost_log

    return df

def calculate_metrics(returns: pd.Series, n_bars_in_year: int = 252) -> dict:
    r = returns.dropna()
    
    gross_wins = r[r > 0].sum()
    gross_losses = r[r < 0].abs().sum()
    profit_factor = gross_wins / gross_losses if gross_losses != 0 else np.nan
    
    annualized_log_return = r.mean() * n_bars_in_year
    annualized_percentage_return = np.exp(annualized_log_return) - 1
    
    sharpe_ratio = (r.mean() / r.std()) * (n_bars_in_year ** 0.5) if r.std() != 0 else 0.0
    
    downside_standard_deviation = r[r < 0].std()
    sortino_ratio = (r.mean() / downside_standard_deviation) * (n_bars_in_year ** 0.5) if downside_standard_deviation != 0 else 0.0
    
    equity_curve = r.cumsum()
    historical_peak = equity_curve.expanding().max()
    drawdown_from_peak = equity_curve - historical_peak
    maximum_drawdown = drawdown_from_peak.min()
    
    return {
        "Annualized Return": annualized_percentage_return,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Profit Factor": profit_factor,
        "Max Drawdown": maximum_drawdown
    }

def main():
    os.makedirs('reports', exist_ok=True)
    
    with open('config.yaml') as f: 
        config = yaml.safe_load(f)
        
    # Dynamically split data into IS and OOS
    is_df, oos_df = load_and_split(config['data']['file_path'], config['oos_settings']['oos_months'])
    fx_params = config['fx_params']
    
    # 1. Process IS Period
    print("Calculating Buy & Hold Benchmark for In-Sample Period...")
    is_bh_results = buy_and_hold_returns(is_df[['open', 'high', 'low', 'close']], **fx_params)
    is_bh_metrics = calculate_metrics(is_bh_results["daily_strategy_return"])
    
    # 2. Process OOS Period
    print("Calculating Buy & Hold Benchmark for Out-of-Sample Period...")
    oos_bh_results = buy_and_hold_returns(oos_df[['open', 'high', 'low', 'close']], **fx_params)
    oos_bh_metrics = calculate_metrics(oos_bh_results["daily_strategy_return"])
    
    # Summary
    report = f"""## Buy and Hold Benchmark (In-Sample Period)
- **Annualized Return:** {is_bh_metrics['Annualized Return'] * 100:.2f}%
- **Sharpe Ratio:** {is_bh_metrics['Sharpe Ratio']:.4f}
- **Sortino Ratio:** {is_bh_metrics['Sortino Ratio']:.4f}
- **Profit Factor:** {is_bh_metrics['Profit Factor']:.4f}
- **Max Drawdown:** {is_bh_metrics['Max Drawdown']:.4f}

## Buy and Hold Benchmark (Out-of-Sample Period)
- **Annualized Return:** {oos_bh_metrics['Annualized Return'] * 100:.2f}%
- **Sharpe Ratio:** {oos_bh_metrics['Sharpe Ratio']:.4f}
- **Sortino Ratio:** {oos_bh_metrics['Sortino Ratio']:.4f}
- **Profit Factor:** {oos_bh_metrics['Profit Factor']:.4f}
- **Max Drawdown:** {oos_bh_metrics['Max Drawdown']:.4f}
"""
    report_path = 'reports/benchmark_summary.md'
    with open(report_path, 'w') as f:
        f.write(report)
        
    print(f"\nBenchmark summary saved to: {report_path}")

if __name__ == "__main__":
    main()