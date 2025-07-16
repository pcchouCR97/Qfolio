import matplotlib.pyplot as plt
import numpy as np 

def StatisticCalculatorRolling(data, freq, print_out = False, plot = False):
    # Compute monthly returns (percentage change from last month's closing price)
    returns = data.resample(freq).last().pct_change().dropna()
    
    # Compute mean return per asset (expected return vector)
    r_i = returns.mean().values

    # Compute covariance matrix of returns
    sigma = returns.cov().values

    if print_out == True:
        # Display results
        print("Mean Period Returns Vector (r_i):\n", r_i)
        print("\nCovariance Matrix (sigma):\n", sigma)

    if plot == True:
        # plot sigma (covarience matrix)
        plt.imshow(sigma, interpolation="nearest")
        plt.show()

    return returns, r_i, sigma

def StatisticCalculator(data, freq, print_out = True, plot = True):
    # Compute monthly returns (percentage change from last month's closing price)
    returns = data.resample(freq).last().pct_change().dropna()

    # Compute mean return per asset (expected return vector)
    r_i = returns.mean().values
    
    # Compute covariance matrix of returns
    sigma = returns.cov().values

    if print_out == True:
        # Display results
        print("Mean Period Returns Vector (r_i):\n", r_i)
        print("\nCovariance Matrix (sigma):\n", sigma)

    if plot == True:
        # plot sigma (covarience matrix)
        plt.imshow(sigma, interpolation="nearest")
        plt.show()

    return returns, r_i, sigma

def Estimate_num_bits(budget, prices):
    min_price = min(prices)  # Find the lowest stock price
    num_bits = int(np.ceil(np.log2(budget / min_price)))  # Compute required bits
    return num_bits

def estimate_num_bits_basek(b: float, p: list[float], k: int) -> int:
    """
    Estimate the number of bits required to encode a value using base-k encoding.

    Args:
        b (float): The budget or maximum reference value.
        p (list[float]): A list of stock prices.
        k (int): The base for encoding.

    Returns:
        int: The number of bits required for base-k encoding.
    """
    min_price = min(p)  # Find the lowest stock price
    ratio = b / min_price  # Compute the ratio of budget to minimum price
    num_bits = int(np.ceil(np.emath.logn(k, ratio)))  # Calculate bits needed using base-k log
    return num_bits

def StatisticCalculatorRollingD(data, train_start, train_end, freq, print_out = False, plot = False):
    # Compute monthly returns (percentage change from last month's closing price)
    returns = data.loc[train_start:train_end].pct_change().dropna()
    #print(returns)
    # Compute mean return per asset (expected return vector)
    r_i = returns.mean().values

    # Compute covariance matrix of returns
    sigma = returns.cov().values

    if print_out == True:
        # Display results
        print("Mean Period Returns Vector (r_i):\n", r_i)
        print("\nCovariance Matrix (sigma):\n", sigma)

    if plot == True:
        # plot sigma (covarience matrix)
        plt.imshow(sigma, interpolation="nearest")
        plt.show()

    return returns, r_i, sigma


def save_files():

    return None