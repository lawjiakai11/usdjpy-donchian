import numpy as np
import pandas as pd
from src.strategy import donchian_breakout
from src.backtest import signal_returns, performance_metrics

def permute_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    log_open = np.log(df['open'].values)
    log_close = np.log(df['close'].values)
    
    # 1. Intrabar Shape Extraction
    relative_high = np.log(df['high'].values) - log_open
    relative_low = np.log(df['low'].values) - log_open
    relative_close = log_close - log_open
    
    # 2. Gap Extraction
    overnight_gaps = np.zeros(len(df))
    overnight_gaps[1:] = log_open[1:] - log_close[:-1]
    
    # FX 24/5 Session Nuance: Explicitly separate weekend gaps (Friday close -> Monday open)
    day_of_week = df.index.dayofweek.to_numpy()
    is_monday = (day_of_week == 0)
    is_monday[0] = False  # Prevent boundary error on first row
    
    weekday_gaps = overnight_gaps[~is_monday]
    weekend_gaps = overnight_gaps[is_monday]
    
    # Shuffle intraday shapes, weekday gaps, and weekend gaps independently
    shape_shuffle_idx = np.random.permutation(len(df))
    weekday_gap_shuffle_idx = np.random.permutation(len(weekday_gaps))
    weekend_gap_shuffle_idx = np.random.permutation(len(weekend_gaps))
    
    shuffled_relative_high = relative_high[shape_shuffle_idx]
    shuffled_relative_low = relative_low[shape_shuffle_idx]
    shuffled_relative_close = relative_close[shape_shuffle_idx]
    
    # Reconstruct synthetic gaps
    synthetic_gaps = np.zeros(len(df))
    synthetic_gaps[~is_monday] = weekday_gaps[weekday_gap_shuffle_idx]
    synthetic_gaps[is_monday] = weekend_gaps[weekend_gap_shuffle_idx]
    synthetic_gaps[0] = 0.0
    
    # Reconstruct synthetic OHLC in log space
    synthetic_log_open = np.zeros(len(df))
    synthetic_log_close = np.zeros(len(df))
    synthetic_log_open[0] = log_open[0]
    synthetic_log_close[0] = synthetic_log_open[0] + shuffled_relative_close[0]
    
    if len(df) > 1:
        synthetic_log_close[1:] = synthetic_log_close[0] + np.cumsum(synthetic_gaps[1:] + shuffled_relative_close[1:])
        synthetic_log_open[1:] = synthetic_log_close[:-1] + synthetic_gaps[1:]
        
    synthetic_log_high = synthetic_log_open + shuffled_relative_high
    synthetic_log_low = synthetic_log_open + shuffled_relative_low
    
    # Convert back to normal price space
    return pd.DataFrame({
        'open': np.exp(synthetic_log_open), 'high': np.exp(synthetic_log_high),
        'low': np.exp(synthetic_log_low), 'close': np.exp(synthetic_log_close)
    }, index=df.index)

def optimize_lookback(df: pd.DataFrame, grid: list, fee_params: dict, return_res: bool = False) -> tuple:
    best_sharpe_ratio = -np.inf
    best_lookback = grid[0]
    best_results = None
    
    for current_lookback in grid:
        current_results = donchian_breakout(df.copy(), current_lookback)
        current_results = signal_returns(current_results, **fee_params)
        current_metrics = performance_metrics(current_results)
        
        if current_metrics["Sharpe Ratio"] > best_sharpe_ratio:
            best_sharpe_ratio = current_metrics["Sharpe Ratio"]
            best_lookback = current_lookback
            best_results = current_results
            
    if return_res:
        return best_lookback, best_sharpe_ratio, best_results
    return best_lookback, best_sharpe_ratio

def walk_forward(df_is: pd.DataFrame, df_oos: pd.DataFrame, grid: list, fee_params: dict) -> pd.DataFrame:
    training_window = df_is[['open', 'high', 'low', 'close']].copy()
    monthly_results_list = []
    
    for month in df_oos.index.to_period('M').unique():
        
        # 1. Re-optimize N using all available historical data
        optimised_lookback, _ = optimize_lookback(training_window, grid, fee_params)
        
        # 2. Trade the entire month m using N*_m
        current_month_df = df_oos[df_oos.index.to_period('M') == month][['open', 'high', 'low', 'close']].copy()
        combined_data = pd.concat([training_window, current_month_df])
        combined_data = donchian_breakout(combined_data, optimised_lookback)
        combined_data = signal_returns(combined_data, **fee_params)
        
        monthly_results_list.append(combined_data.loc[current_month_df.index])
        
        # 3. Append month m to training dataset (Expanding Window)
        training_window = pd.concat([training_window, current_month_df])
        
    return pd.concat(monthly_results_list)