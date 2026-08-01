# USDJPY Donchian Breakout

While the Donchian breakout is a classic systematic rule, the primary focus of this project is on **statistical rigor**. The pipeline enforces strict look-ahead prevention, models realistic OTC Forex microstructure, and utilises Monte Carlo permutation testing to eliminate data snooping bias.

## Key Features

- **FX Microstructure Modeling:** Replaces naive flat fees with a realistic 0.5 pip ECN spread/commission model and a 2% annualized Cost of Carry (Swap proxy) to account for the USD/JPY interest rate differential.
- **24/5 Session Permutations:** A custom Monte Carlo permutation engine that preserves intrabar volatility, fat tails, and explicitly isolates and shuffles weekend gaps (Friday close $\to$ Monday open) independently from weekday overnight gaps.
- **Expanding Walk-Forward:** Re-optimizes the lookback parameter monthly using an expanding training window, preserving long-term macroeconomic regime context.
- **"Fair Competition" Optimization:** Forces 1,000 synthetic noise datasets through the exact same 6-parameter optimization grid as the real data, generating a mathematically pure null distribution to calculate true p-values.

## Repository Structure

```text
usdjpy-donchian/
│
├── FULL_REPORT.ipynb           # Full research report
├── README.md                   
├── pyproject.toml              
├── config.yaml                 # Parameters (data path, FX costs, grids)
│
├── data/
│   └── raw/
│       └── USDJPY.csv          
│
├── src/
│   ├── data.py                 # CSV ingestion 
│   ├── strategy.py             # Donchian signal generation
│   ├── backtest.py             # FX returns, costs, carry, and metric calculations
│   └── validation.py           # Permutation engine & walk-forward logic
│
├── scripts/
│   └── run_research.py         # Main orchestrator script
│
├── tests/
│   ├── test_core.py            # Tests signal logic and metrics
│   └── test_validation.py      # Tests permutation integrity and walk-forward
│
└── reports/
    ├── figures/                # Histograms & equity curves
    ├── archived/               # Results from my run, used in FULL_REPORT
    └── research_summary.md     # Overview of metrics & p-values
```

## Setup & Execution

### 1. Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
python3 -m pip install -e '.[test]'
```

### 2. Run Tests
```bash
pytest
```

### 3. Execute the Research Pipeline
Run the main script to execute the full optimization, permutation testing, and report generation:
```bash
python -m scripts.run_research
```

## Outputs
Upon completion, the pipeline generates:
1. **`reports/final_report.md`**: A comprehensive markdown file containing IS/OOS performance metrics (Sharpe, Sortino, Profit Factor, Max Drawdown, Annualized Return) and statistical p-values.
2. **`reports/figures/`**: Four visualizations:
   - IS Sharpe Distribution (Real vs. Permutations)
   - IS Equity Curves (Real vs. Permutation cloud)
   - OOS Sharpe Distribution (Real vs. Permutations)
   - OOS Equity Curves (Real vs. Permutation cloud)

## Configuration
All research parameters can be adjusted in `config.yaml`:
```yaml
fx_params:
  pip_cost: 0.5          # Round-turn spread + commission in pips
  point: 0.01            # USDJPY pip size (0.01)
  annual_carry: 0.02     # Approximation: 2% annualized USD/JPY interest rate differential proxy

optimization:
  lookback_grid: [10, 20, 50, 100, 150, 200]

permutation:             
  in_sample_runs: 1000          
  walk_forward_runs: 1000
# If you are trying it out, 1000 permutations will take about 20-30 min lol, though main bottleneck is walk_forward_runs due to expanding window iterations, recommend (1000, 200) for lightweight testing.
```