import numpy as np
import pandas as pd

def donchian_breakout(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    df["upper"] = df["close"].rolling(lookback - 1).max().shift(1)
    df["lower"] = df["close"].rolling(lookback - 1).min().shift(1)
    df["signal"] = np.nan
    df.loc[df["close"] > df["upper"], "signal"] = 1
    df.loc[df["close"] < df["lower"], "signal"] = -1
    df["signal"] = df["signal"].ffill()
    return df