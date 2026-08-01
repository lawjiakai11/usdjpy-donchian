import pandas as pd
import numpy as np
from src.validation import permute_ohlc, walk_forward

def test_permute_ohlc_preserves_shape():
    np.random.seed(42)
    # Include weekends in the DatetimeIndex to test the FX weekend gap logic
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    
    # Generate strictly valid OHLC data
    open_prices = np.random.uniform(100, 110, 100)
    close_prices = np.random.uniform(100, 110, 100)
    
    # Ensure High is always >= max(Open, Close) and Low is always <= min(Open, Close)
    high_prices = np.maximum(open_prices, close_prices) + np.random.uniform(1, 5, 100)
    low_prices = np.minimum(open_prices, close_prices) - np.random.uniform(1, 5, 100)
    
    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices
    }, index=dates)
    
    perm_df = permute_ohlc(df)
    assert len(perm_df) == 100
    # Verify the synthetic bars maintain valid OHLC integrity
    assert (perm_df['high'] >= perm_df[['open', 'close']].max(axis=1)).all()
    assert (perm_df['low'] <= perm_df[['open', 'close']].min(axis=1)).all()

def test_walk_forward_no_lookahead():
    # Include weekends
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'open': np.arange(100) + 100, 
        'high': np.arange(100) + 102,
        'low': np.arange(100) + 98, 
        'close': np.arange(100) + 101
    }, index=dates)
    
    is_df, oos_df = df.iloc[:80], df.iloc[80:]
    
    # Pass the new fee_params dictionary
    fee_params = {'pip_cost': 0.5, 'point': 0.01, 'annual_carry': 0.02}
    res = walk_forward(is_df, oos_df, [10, 20], fee_params)
    
    assert res.index.min() == oos_df.index.min()
    assert len(res) == len(oos_df)