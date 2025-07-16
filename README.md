# Qfolio

Qfolio is a portfolio optimization research package that benchmarks classical and quantum solvers under realistic market conditions. The project compares a NumPy-based classical optimizer and the Quantum Approximate Optimization Algorithm (QAOA), implemented using [Qiskit](https://qiskit.org/), to assess their performance across bull and bear market regimes.

## Features

-   Classical and quantum optimization using QUBO formulation
-   Hamiltonian-based modeling of return and risk trade-offs
-   Binary encoding for integer share allocation
-   Support for rebalancing with and without periodic investment
-   Bull and bear market case studies using real VOO data
-   Visualized return comparisons for AAPL, MSFT, AMZN, and VOO
-   Integration with Qiskit’s `MinimumEigenOptimizer`, `QAOA`, and `COBYLA`

## Background

Portfolio optimization is a foundational challenge in finance, with classical methods often struggling under computational complexity in combinatorial cases. Qfolio reformulates the portfolio selection task into a Quadratic Unconstrained Binary Optimization (QUBO) problem and explores how quantum algorithms like QAOA can enhance solution quality and efficiency.

The methodology draws upon:

-   Hamiltonian formulations of return and risk
-   Binary encoding to enable QUBO compliance
-   Qiskit’s classical and quantum solvers
-   Benchmarks based on real S&P 500 data (via the VOO ETF)

## Example

A <div style="text-align: center;">
    <img src="../../paper/figures/_VOO_bb_bull_study_post.png" alt="_VOO_bb_bull_study_post" style="width: 1050px; height: auto;">
    <p style="font-size: 16px; font-style: italic; color: gray; margin-top: 5px;">
        Return \( \% \) for AAPL, MSFT, AMZN, and VOO versus optimized portfolios from 2023-10-27 to 2024-10-24 (bull market). The optimization used \( \lambda_{1} = 1000 \) with an initial \( 10,000 \)investment, rebalanced every 21 days without additional contributions.
    </p>
</div>

Full paper link: [full paper](https://github.com/pcchouCR97/Qfolio/blob/main/paper/paper.pdf)


## References

-   [Ahmed (2002)](https://www2.isye.gatech.edu/~shabbir/ISyE6669/)
-   [D-Wave Portfolio Optimization](https://github.com/dwave-examples/portfolio-optimization)
-   [Qiskit Finance Portfolio Optimization](https://qiskit-community.github.io/qiskit-finance tutorials/01_portfolio_optimization.html)
-   [Vanguard VOO ETF](https://investor.vanguard.com/investment-products/etfs/profile/voo)
