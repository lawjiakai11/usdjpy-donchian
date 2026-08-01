import pandas as pd
import numpy as np
from src.strategy import donchian_breakout
from src.backtest import signal_returns, performance_metrics

def test_donchian_logic():
    df = pd.DataFrame({
        'open': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'high': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'low': [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5],
        'close': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })
    res = donchian_breakout(df.copy(), 5)
    assert res.loc[4, 'signal'] == 1  # 5 > 4 (max of 1..4 shifted)
    assert pd.isna(res.loc[3, 'signal'])

def test_returns_and_metrics():
    df = pd.DataFrame({
        'open': [10, 11, 12, 11, 10, 9, 10, 11, 12, 13],
        'high': [11, 12, 13, 12, 11, 10, 11, 12, 13, 14],
        'low': [9, 10, 11, 10, 9, 8, 9, 10, 11, 12],
        'close': [10, 11, 12, 11, 10, 9, 10, 11, 12, 13]
    })
    res = donchian_breakout(df, 3)
    
    # Pass the new FX parameters
    fee_params = {'pip_cost': 0.5, 'point': 0.01, 'annual_carry': 0.02}
    res = signal_returns(res, **fee_params)
    
    metrics = performance_metrics(res)
    assert 'Sharpe Ratio' in metrics
    assert 'Max Drawdown' in metrics
    assert 'Annualized Return' in metrics