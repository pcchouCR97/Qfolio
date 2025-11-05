# Qfolio

> Quantum-enhanced portfolio optimization with AMSP encoding and JPMorgan decomposition

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](https://github.com/pcchouCR97/Qfolio/blob/main/paper/paper.pdf)

Qfolio is a portfolio optimization research package that benchmarks classical and quantum solvers under realistic market conditions. The project compares NumPy-based classical optimizers and the Quantum Approximate Optimization Algorithm (QAOA), implemented using [Qiskit](https://qiskit.org/), to assess their performance across bull and bear market regimes.

**[Quick Start](#quick-start) | [Examples](#examples--tutorials) | [Features](#features) | [Paper](https://github.com/pcchouCR97/Qfolio/blob/main/paper/paper.pdf) | [Citation](#citation)**

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/pcchouCR97/Qfolio.git
cd Qfolio

# Create virtual environment
python -m venv qfolio_dev
source qfolio_dev/bin/activate  # Linux/Mac
qfolio_dev\Scripts\activate     # Windows

# Install package
pip install -e .
```

### 5-Minute Example

```python
# Run JPMorgan decomposition optimization
from qfolio.optimization.jpmorgan_pipeline import run_jpmorgan_pipeline
from qfolio.screeners.sharpe_screener import load_data

# Load S&P 500 sample data (42 stocks)
data = load_data("examples/example_csv/SP500_42stocks_baseline_adjusted_close_103025.csv")

# Select assets
tickers = ['AAPL', 'MSFT', 'NVDA']
current_prices = data.iloc[-1][tickers].to_dict()

# Run optimization
weights = run_jpmorgan_pipeline(
    prices_data=data,
    tickers=tickers,
    budget=10000,
    current_prices=current_prices,
    quantum_solver='classic',  # or 'QAOA' for quantum
    max_community_size=3
)

print(f"Optimal weights: {weights}")
```

### Run Example Backtests

```bash
# Standard AMSP backtest
python examples/AMSP_example.py

# JPMorgan decomposition backtest
python examples/AMSP_JP_backtest.py
```

---

## Examples & Tutorials

### Python Scripts

Run complete backtests from the command line:

#### 1. Standard AMSP Backtest
```bash
python examples/AMSP_example.py
```
Compare classical (NumPy) vs quantum (QAOA) solvers on multi-asset portfolio with AMSP encoding. Generates QuantStats HTML report with 50+ performance metrics.

**Key Features:**
- AMSP binary encoding for efficient QUBO representation
- Classical vs quantum solver comparison
- Forward-looking backtest with periodic rebalancing
- Comprehensive QuantStats tearsheet

#### 2. JPMorgan Decomposition Pipeline
```bash
python examples/AMSP_JP_backtest.py
```
Large-scale portfolio optimization using Random Matrix Theory preprocessing and spectral clustering. Demonstrates decomposition for 42+ stock portfolios.

**Key Features:**
- RMT (Marchenko-Pastur) noise filtering
- Newman's spectral bisection clustering
- Per-community optimization and aggregation
- Scalable to large universes

#### 3. Backward-Looking Analysis
```bash
python examples/AMSP_example_backward_looking.py
```
Historical performance analysis mode for research and validation studies. Uses complete dataset for retrospective analysis.

---

### Interactive Jupyter Tutorials

Explore step-by-step guides in [`examples/tutorial/`](examples/tutorial/):

#### [Qfolio_VOO_Benchmark_Tutorial.ipynb](examples/tutorial/Qfolio_VOO_Benchmark_Tutorial.ipynb)
- Compare portfolio performance vs VOO ETF benchmark
- Understand risk-return trade-offs
- Visualize cumulative returns and drawdowns
- **Perfect for beginners**

**What you'll learn:**
- Loading and preprocessing stock data
- Running optimization with different parameters
- Interpreting performance metrics
- Benchmarking against market indices

#### [Qfolio_AMSP_42_Stocks_Tutorial.ipynb](examples/tutorial/Qfolio_AMSP_42_Stocks_Tutorial.ipynb)
- Walk through 42-stock S&P 500 portfolio optimization
- Deep dive into AMSP encoding mechanics
- JPMorgan decomposition methodology
- Advanced parameter tuning

**What you'll learn:**
- AMSP encoding: k^0 + k^1 + k^2 + ... formula
- RMT preprocessing and eigenvalue filtering
- Spectral clustering for community detection
- Risk parameter (q) and budget penalty (λ₁) tuning

---

## Features

### Optimization Algorithms
- **Classical Solvers**: NumPy, CPLEX, Gurobi, SCIP
- **Quantum Solver**: QAOA (Quantum Approximate Optimization Algorithm)
- **Hybrid Approach**: JPMorgan RMT + Spectral Clustering decomposition

### Advanced Techniques
- **AMSP Encoding**: Adaptive Multi-Scale Pricing for efficient binary representation of portfolio positions
- **JPMorgan Pipeline**: Random Matrix Theory preprocessing + Newman's spectral bisection clustering
- **Risk Management**: CVaR optimization, Sharpe ratio screening, dynamic asset universe
- **Market Calendar Integration**: NYSE trading days with `pandas_market_calendars`

### Analysis & Reporting
- **QuantStats Integration**: Professional tearsheet reports with 50+ performance metrics
- **Performance Metrics**: Sharpe ratio, Sortino ratio, max drawdown, CVaR, rolling volatility
- **Bull/Bear Case Studies**: Real VOO ETF data (2023-2024) with market regime analysis
- **Visualization**: Return comparisons, allocation evolution, risk-return profiles

### Backtesting Framework
- Forward-looking backtests with no look-ahead bias
- Periodic rebalancing with configurable frequency (daily/monthly)
- Dynamic asset universe with eligibility tracking
- Transaction cost modeling (optional)
- Consistent returns calculation with `StatisticCalculatorRollingD`

---

## Analysis & Reporting

### QuantStats Integration

Automatically generate professional performance tearsheets:

```bash
python examples/AMSP_JP_backtest.py
# Output: quantstats_reports_JP/portfolio_report_[dates].html
```

**50+ Metrics Included:**
- Cumulative returns vs benchmark (SPY/VOO)
- Sharpe ratio, Sortino ratio, Calmar ratio
- Maximum drawdown, volatility analysis
- Value at Risk (VaR), Conditional VaR (CVaR)
- Beta, R-squared vs benchmark
- Monthly returns heatmap
- Rolling performance charts (Sharpe, volatility, beta)
- Distribution analysis and tail risk

Reports saved to [`quantstats_reports_JP/`](quantstats_reports_JP/) directory. Open HTML files in any browser for interactive exploration.

---

## Results

![Bull Market Returns](https://raw.githubusercontent.com/pcchouCR97/Qfolio/main/paper/figures/_VOO_bb_bull_study_post.png)

*Return (%) for AAPL, MSFT, AMZN, and VOO versus optimized portfolios from 2023-10-27 to 2024-10-24 (bull market). The optimization used λ₁ = 1000 with an initial $10,000 investment, rebalanced every 21 days without additional contributions.*

Full paper: **[View PDF](https://github.com/pcchouCR97/Qfolio/blob/main/paper/paper.pdf)**

---

## Background

Portfolio optimization is a foundational challenge in finance, with classical methods often struggling under computational complexity in combinatorial cases. Qfolio reformulates the portfolio selection task into a Quadratic Unconstrained Binary Optimization (QUBO) problem and explores how quantum algorithms like QAOA can enhance solution quality and efficiency.

The methodology draws upon:
- Hamiltonian formulations of return and risk trade-offs
- Binary encoding to enable QUBO compliance
- Qiskit's classical and quantum solvers
- Benchmarks based on real S&P 500 data (via the VOO ETF)

---

## Package Structure

```
qfolio/
├── optimization/          # Portfolio optimization algorithms
│   ├── jpmorgan_pipeline.py    # RMT + Spectral Clustering
│   ├── hamiltonian.py          # QUBO Hamiltonian formulation
│   ├── amsp_encoder.py         # AMSP encoding utilities
│   └── core.py                 # Backtest orchestration
├── screeners/            # Asset screening
│   └── sharpe_screener.py      # Sharpe ratio-based selection
├── backtesting/          # Backtesting framework
│   ├── PortfolioManager_AMSP.py
│   ├── date_loader.py          # Market calendar utils
│   └── portfolio.py            # Portfolio analytics
├── data/                 # Data management
│   └── DataManager.py          # CSV/API data loading
├── analysis/             # Performance analysis
│   ├── comparison.py           # Solver comparison
│   └── metrics.py              # Performance calculations
├── metrics/              # Risk metrics
│   └── RiskMetrics.py          # CVaR, VaR, etc.
├── results/              # Results management
│   └── exporter.py             # Save backtest results
└── visualization/        # Plotting utilities
    └── plotter.py              # Portfolio visualizations

examples/                 # Example backtests & tutorials
├── AMSP_example.py              # Standard AMSP backtest
├── AMSP_JP_backtest.py          # JPMorgan pipeline backtest
├── AMSP_example_backward_looking.py  # Historical analysis
├── example_csv/                 # Sample S&P 500 data
└── tutorial/                    # Jupyter notebooks
    ├── Qfolio_VOO_Benchmark_Tutorial.ipynb
    └── Qfolio_AMSP_42_Stocks_Tutorial.ipynb
```

---

## Documentation

### Python API

```python
# JPMorgan Pipeline
from qfolio.optimization.jpmorgan_pipeline import run_jpmorgan_pipeline

# Sharpe Screening
from qfolio.screeners.sharpe_screener import SharpeRatioCalculator

# Risk Metrics
from qfolio.metrics.RiskMetrics import RiskMetricsCalculator

# Data Management
from qfolio.data.DataManager import DataManager

# Backtesting Core
from qfolio.optimization.core import BackTrackCore, OptimizationConfig
```

### CLI Commands

```bash
# Compare classical vs quantum solvers
qfolio-compare --config config.yaml

# Run backtest from CLI
qfolio-backtest --data data.csv --budget 10000 --solver QAOA
```

---

## Contributing

We welcome contributions! Whether you're fixing bugs, adding features, or improving documentation, your help is appreciated.

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install with dev dependencies
pip install -e .[dev]

# Run tests
pytest tests/

# Format code
black qfolio/

# Type checking
mypy qfolio/
```

---

## Citation

If you use Qfolio in your research, please cite:

```bibtex
@article{chou2024qfolio,
  title={Qfolio: Quantum-Enhanced Portfolio Optimization with AMSP Encoding},
  author={Chou, Po-Chih},
  journal={GitHub Repository},
  year={2024},
  url={https://github.com/pcchouCR97/Qfolio}
}
```

---

## References

- [Ahmed (2002) - Portfolio Optimization](https://www2.isye.gatech.edu/~shabbir/ISyE6669/)
- [D-Wave Portfolio Optimization Examples](https://github.com/dwave-examples/portfolio-optimization)
- [Qiskit Finance Portfolio Optimization Tutorial](https://qiskit-community.github.io/qiskit-finance/tutorials/01_portfolio_optimization.html)
- [Vanguard VOO ETF Data](https://investor.vanguard.com/investment-products/etfs/profile/voo)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Built with [Qiskit](https://qiskit.org/) for quantum computing
- Market data from [Yahoo Finance](https://finance.yahoo.com/) via `yfinance`
- Performance analysis powered by [QuantStats](https://github.com/ranaroussi/quantstats)
- Optimization solvers: CPLEX, Gurobi, SCIP

---

**Questions?** Open an [issue](https://github.com/pcchouCR97/Qfolio/issues) or reach out to [pcchouCR97@gmail.com](mailto:pcchouCR97@gmail.com)
