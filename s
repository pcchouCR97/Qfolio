warning: in the working copy of '.gitignore', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'examples/AMSP_example_explore_regular.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'examples/AMSP_example_hybrid.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'examples/tutorial/Qfolio_AMSP_42_Stocks_Tutorial.ipynb', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'examples/tutorial/Qfolio_VOO_Benchmark_Tutorial.ipynb', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'qfolio/backtesting/PortfolioOptimizer_AMSP.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'qfolio/optimization/hamiltonian.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'qfolio/optimization/jpmorgan_pipeline.py', LF will be replaced by CRLF the next time Git touches it
[1mdiff --git a/.gitignore b/.gitignore[m
[1mindex cc01ac28..d153e83c 100644[m
[1m--- a/.gitignore[m
[1m+++ b/.gitignore[m
[36m@@ -77,4 +77,7 @@[m [mhtmlcov/[m
 dmypy.json[m
 [m
 # claude[m
[31m-.claude/[m
\ No newline at end of file[m
[32m+[m[32m.claude/[m
[32m+[m
[32m+[m[32m# Test files[m
[32m+[m[32mtest.py[m
\ No newline at end of file[m
[1mdiff --git a/README.md b/README.md[m
[1mindex c04ee853..0e65c0be 100644[m
[1m--- a/README.md[m
[1m+++ b/README.md[m
[36m@@ -1,14 +1,14 @@[m
 # Qfolio[m
 [m
[31m-> Quantum-enhanced portfolio optimization with AMSP encoding and JPMorgan decomposition[m
[32m+[m[32m> Quantum-enhanced portfolio optimization with AMSP encoding and Spectral Clustering decomposition[m
 [m
 [![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)[m
 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)[m
[31m-[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](https://github.com/pcchouCR97/Qfolio/blob/main/paper/paper.pdf)[m
[32m+[m[32m[![Reprot](https://img.shields.io/badge/Paper-PDF-red.svg)](https://github.com/pcchouCR97/Qfolio/blob/main/paper/paper.pdf)[m
 [m
[31m-Qfolio is a portfolio optimization research package that benchmarks classical and quantum solvers under realistic market conditions. The project compares NumPy-based classical optimizers and the Quantum Approximate Optimization Algorithm (QAOA), implemented using [Qiskit](https://qiskit.org/), to assess their performance across bull and bear market regimes.[m
[32m+[m[32mQfolio is a portfolio optimization research package that benchmarks classical and quantum solvers under realistic market conditions. The project compares NumPy-based classical optimizers and the Quantum Approximate Optimization Algorithm (QAOA), Variational Quantum Eigensolver (VQE), implemented using [Qiskit](https://qiskit.org/), to assess their performance across bull and bear market regimes.[m
 [m
[31m-**[Quick Start](#quick-start) | [Examples](#examples--tutorials) | [Features](#features) | [Paper](https://github.com/pcchouCR97/Qfolio/blob/main/paper/paper.pdf) | [Citation](#citation)**[m
[32m+[m[32m**[Quick Start](#quick-start) | [Examples](#examples--tutorials) | [Features](#features) | [Report](https://github.com/pcchouCR97/Qfolio/blob/main/paper/paper.pdf) | [⭐ Star](https://github.com/pcchouCR97/Qfolio) | [Citation](#citation)**[m
 [m
 ---[m
 [m
[36m@@ -33,7 +33,7 @@[m [mpip install -e .[m
 ### 5-Minute Example[m
 [m
 ```python[m
[31m-# Run JPMorgan decomposition optimization[m
[32m+[m[32m# Run Spectral Clustering decomposition optimization[m
 from qfolio.optimization.jpmorgan_pipeline import run_jpmorgan_pipeline[m
 from qfolio.screeners.sharpe_screener import load_data[m
 [m
[36m@@ -41,8 +41,8 @@[m [mfrom qfolio.screeners.sharpe_screener import load_data[m
 data = load_data("examples/example_csv/SP500_42stocks_baseline_adjusted_close_103025.csv")[m
 [m
 # Select assets[m
[31m-tickers = ['AAPL', 'MSFT', 'NVDA'][m
[31m-current_prices = data.iloc[-1][tickers].to_dict()[m
[32m+[m[32mtickers = ['AAPL', 'MSFT', 'NVDA','AVGO','ORCL','V'][m
[32m+[m[32mcurrent_prices = data.iloc[-1][tickers].values[m
 [m
 # Run optimization[m
 weights = run_jpmorgan_pipeline([m
[36m@@ -50,21 +50,30 @@[m [mweights = run_jpmorgan_pipeline([m
     tickers=tickers,[m
     budget=10000,[m
     current_prices=current_prices,[m
[31m-    quantum_solver='classic',  # or 'QAOA' for quantum[m
[32m+[m[32m    quantum_solver='classic',  # or 'QAOA', 'SamplerVQE' for quantum algorithm[m
     max_community_size=3[m
 )[m
[31m-[m
[32m+[m[32mprint(tickers)[m
 print(f"Optimal weights: {weights}")[m
 ```[m
 [m
 ### Run Example Backtests[m
 [m
 ```bash[m
[31m-# Standard AMSP backtest[m
[32m+[m[32m# To-date backtest - Run backtest up to a specific date[m
[32m+[m[32mpython examples/AMSP_example_backward_looking.py[m
[32m+[m
[32m+[m[32m# Standard AMSP backtest - Basic portfolio optimization with rebalancing[m
 python examples/AMSP_example.py[m
 [m
[31m-# JPMorgan decomposition backtest[m
[32m+[m[32m# Spectral Clustering decomposition backtest - Uses decomposition pipeline with community detection[m
 python examples/AMSP_JP_backtest.py[m
[32m+[m
[32m+[m[32m# Magnificent 7 backtest - Test on tech giants[m
[32m+[m[32mpython examples/AMSP_testbed_Mag7.py[m
[32m+[m
[32m+[m[32m# Grid study - Parameter exploration across multiple configurations[m
[32m+[m[32mpython examples/AMSP_example_explore_regular.py[m
 ```[m
 [m
 ---[m
[36m@@ -82,12 +91,18 @@[m [mpython examples/AMSP_example.py[m
 Compare classical (NumPy) vs quantum (QAOA) solvers on multi-asset portfolio with AMSP encoding. Generates QuantStats HTML report with 50+ performance metrics.[m
 [m
 **Key Features:**[m
[31m-- AMSP binary encoding for efficient QUBO representation[m
[31m-- Classical vs quantum solver comparison[m
[32m+[m[32m- AMSP binary encoding for QUBO representation[m
[32m+[m[32m- Supporting fraction shares[m
 - Forward-looking backtest with periodic rebalancing[m
 - Comprehensive QuantStats tearsheet[m
 [m
[31m-#### 2. JPMorgan Decomposition Pipeline[m
[32m+[m[32m**Results QuantStats example**[m
[32m+[m
[32m+[m[32m<img src="quantstats_reports\result_example01.png" width="800">[m
[32m+[m
[32m+[m[32mFull report: <a href="quantstats_reports/portfolio_report_2020-07-01_2025-10-28.html">View Portfolio Report</a>[m
[32m+[m
[32m+[m[32m#### 2. Spectral Clustering Decomposition Pipeline[m
 ```bash[m
 python examples/AMSP_JP_backtest.py[m
 ```[m
[36m@@ -99,6 +114,14 @@[m [mLarge-scale portfolio optimization using Random Matrix Theory preprocessing and[m
 - Per-community optimization and aggregation[m
 - Scalable to large universes[m
 [m
[32m+[m[32m**Results QuantStats example**[m
[32m+[m
[32m+[m[32m<img src="quantstats_reports_JP/result_example01.png" width="800">[m
[32m+[m
[32m+[m[32mFull report: <a href="quantstats_reports_JP\portfolio_report_JP_2020-07-01_2025-10-28.html">View Portfolio Report</a>[m
[32m+[m
[32m+[m
[32m+[m
 #### 3. Backward-Looking Analysis[m
 ```bash[m
 python examples/AMSP_example_backward_looking.py[m
[36m@@ -113,40 +136,33 @@[m [mExplore step-by-step guides in [`examples/tutorial/`](examples/tutorial/):[m
 [m
 #### [Qfolio_VOO_Benchmark_Tutorial.ipynb](examples/tutorial/Qfolio_VOO_Benchmark_Tutorial.ipynb)[m
 - Compare portfolio performance vs VOO ETF benchmark[m
[31m-- Understand risk-return trade-offs[m
 - Visualize cumulative returns and drawdowns[m
[31m-- **Perfect for beginners**[m
 [m
 **What you'll learn:**[m
 - Loading and preprocessing stock data[m
[31m-- Running optimization with different parameters[m
[31m-- Interpreting performance metrics[m
[31m-- Benchmarking against market indices[m
[32m+[m[32m- Running optimization with classical and quantum solvers on your local machine[m
 [m
 #### [Qfolio_AMSP_42_Stocks_Tu