## Data & Split
- **Dataset:** `data/raw/USDJPY.csv`
- **In-Sample Period:** 1991-01-02 to 2011-12-29
- **Out-of-Sample Period:** 2011-12-30 to 2016-12-30

## FX Microstructure Parameters
- **Execution:** 5:00 PM NY Rollover
- **Transaction Cost:** 0.5 pips round-turn
- **Carry Proxy:** 2.0% annualized interest rate differential

## Optimization Results
- **Optimized Lookback (IS):** 200 days
- **In-Sample Annualized Return:** 4.30%
- **In-Sample Sharpe Ratio:** 0.3751
- **In-Sample Sortino Ratio:** 0.5055
- **In-Sample Profit Factor:** 1.0683
- **In-Sample Max Drawdown:** -0.2920

## Out-of-Sample Performance
- **OOS Annualized Return:** 7.04%
- **OOS Sharpe Ratio:** 0.6947
- **OOS Sortino Ratio:** 1.0434
- **OOS Profit Factor:** 1.1312
- **OOS Max Drawdown:** -0.1512

## Permutation Tests (Weekend-Preserving)
- **In-Sample P-Value:** 0.1380 (Null: Sharpe from 1000 permuted datasets >= real IS Sharpe)
- **Walk-Forward P-Value:** 0.0480 (Null: Sharpe from 1000 permuted WF runs >= real OOS Sharpe)

## Conclusion
The strategy exhibits insufficient evidence of persistent momentum after accounting for multiple comparisons, FX spreads, and swap carry.
