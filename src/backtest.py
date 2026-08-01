import numpy as np
import pandas as pd

def signal_returns(price_data: pd.DataFrame, pip_cost: float = 0.5, point: float = 0.01, annual_carry: float = 0.02) -> pd.DataFrame:
    # 1. Strategy return from 5PM NY rollover
    price_data["daily_log_return"] = np.log(price_data["close"]).diff().shift(-1)
    price_data["daily_strategy_return"] = price_data["signal"] * price_data["daily_log_return"]

    # 2. FX Transaction Cost
    pip_cost_log_percentage = np.log(1 - (pip_cost * point) / price_data["close"])
    absolute_position_change = price_data["signal"].diff().abs()
    price_data["daily_strategy_return"] -= absolute_position_change * pip_cost_log_percentage

    # 3. Cost of Carry
    daily_carry_log_percentage = np.log(1 + annual_carry) / 252
    price_data["daily_strategy_return"] += price_data["signal"] * daily_carry_log_percentage

    # Final Net Equity Curve
    price_data["cumulative_equity_curve"] = price_data["daily_strategy_return"].cumsum()
    return price_data

def performance_metrics(price_data: pd.DataFrame, n_bars_in_year: int = 252) -> dict:
    daily_strategy_returns = price_data["daily_strategy_return"].dropna()
    
    gross_wins = daily_strategy_returns[daily_strategy_returns > 0].sum()
    gross_losses = daily_strategy_returns[daily_strategy_returns < 0].abs().sum()
    profit_factor = gross_wins / gross_losses if gross_losses != 0 else np.nan
    
    # Annualized Return (converted from log to percentage)
    annualized_log_return = daily_strategy_returns.mean() * n_bars_in_year
    annualized_percentage_return = np.exp(annualized_log_return) - 1
    
    sharpe_ratio = (daily_strategy_returns.mean() / daily_strategy_returns.std()) * (n_bars_in_year ** 0.5) if daily_strategy_returns.std() != 0 else 0.0
    
    downside_standard_deviation = daily_strategy_returns[daily_strategy_returns < 0].std()
    sortino_ratio = (daily_strategy_returns.mean() / downside_standard_deviation) * (n_bars_in_year ** 0.5) if downside_standard_deviation != 0 else 0.0
    
    equity_curve = daily_strategy_returns.cumsum()
    historical_peak = equity_curve.expanding().max()
    drawdown_from_peak = equity_curve - historical_peak
    maximum_drawdown = drawdown_from_peak.min()
    
    return {
        "Profit Factor": profit_factor,
        "Annualized Return": annualized_percentage_return,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Max Drawdown": maximum_drawdown
    }