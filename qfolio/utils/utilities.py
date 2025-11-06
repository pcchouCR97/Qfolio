import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd


def StatisticCalculatorRolling(data, freq, print_out = False, plot = False):
    """
    # To be deprecated
    """
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

def AMSP_estimator(prices, max_binary_bits, budget, k = 2):
    """
    ASMP: ndarray(len(price))
    lots: ndarray(len(price))
    """
    AMSP = budget / sum(k ** i for i in range(max_binary_bits)) 
    print(AMSP)
    lots = AMSP / prices
    return lots, AMSP

def StatisticCalculatorRollingD(data, train_start, train_end, print_out = False, plot = False):
    """
    This function computes the daily returns, mean return vector, and covariance matrix
    for the given data over the specified training period.

    Args:
        data (pd.DataFrame): DataFrame containing price data with a DatetimeIndex.
        train_start (str or pd.Timestamp): Start date for the training period.
        train_end (str or pd.Timestamp): End date for the training period.
        print_out (bool): Whether to print the results. Default is False.
        plot (bool): Whether to plot the covariance matrix. Default is False.
    Returns:
        returns (pd.DataFrame): DataFrame of daily returns.
        r_i (np.array): Mean return vector.
        sigma (np.array): Covariance matrix of returns. 

    """
    # Compute monthly returns (percentage change from last month's closing price)
    # i.e. day to day return in percent change.
    returns = data.loc[train_start:train_end].pct_change().dropna()
    
    # Compute the mean of the daily return per asset (expected return vector)
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

def StatisticCalculatorRollingD_AMSP(data, train_start, train_end, lots, print_out = False, plot = False):
    
    """
    This function computes the daily returns, mean return vector, and covariance matrix for the given data over the specified training period. WITH AMSP 'per lot' adjustment.

    Args:
        data (pd.DataFrame): DataFrame containing price data with a DatetimeIndex.
        train_start (str or pd.Timestamp): Start date for the training period.
        train_end (str or pd.Timestamp): End date for the training period.

        print_out (bool): Whether to print the results. Default is False.
        plot (bool): Whether to plot the covariance matrix. Default is False.
    Returns:
        returns (pd.DataFrame): DataFrame of daily returns.
        r_i (np.array): Mean return vector.
        sigma (np.array): Covariance matrix of returns. 

    """

    # Compute monthly returns (percentage change from last month's closing price)
    # i.e. day to day return in percent change with AMSP 'per lot' adjustment
    returns = data.loc[train_start:train_end].pct_change().dropna()
    
    # Compute the mean of the daily return per asset (expected return vector) with AMSP 'per lot' adjustment
    r_i = returns.mean().values * lots

    # Compute covariance matrix of returns 'per share'
    sigma_per_share = returns.cov().values
    # then, adjust it to be the coariance 'per lot'
    sigma = sigma_per_share * np.outer(lots, lots)

    if print_out == True:
        # Display results
        #print("Return", returns)
        print("Mean Period Returns Vector (r_i):\n", r_i)
        print("\nCovariance Matrix (sigma):\n", sigma)

    if plot == True:
        # plot sigma (covarience matrix)
        plt.imshow(sigma, interpolation="nearest")
        plt.show()

    return returns, r_i, sigma

def StatisticCalculatorEWMA(data, train_start, train_end, lam=0.97, print_out=False, plot=False):
    # 
    #returns = data.loc[train_start:train_end].pct_change().dropna()
    # log return:
    returns = np.log(data.loc[train_start:train_end].pct_change().dropna() + 1)
    
    T = len(returns)
    weights = np.array([(1-lam)*(lam**k) for k in range(T)][::-1])
    weights /= weights.sum()
    r_i = (returns.values * weights[:, None]).sum(axis=0)

    # EWMA covariance
    m = returns.mean().values
    S = np.eye(returns.shape[1]) * 1e-8
    for r in returns.values:
        u = (r - m)[:, None]
        S = lam*S + (1-lam)*(u @ u.T)
    sigma = S

    if print_out:
        print("EWMA Returns Vector (r_i):\n", r_i)
        print("\nEWMA Covariance Matrix (sigma):\n", sigma)
    if plot:
        plt.imshow(sigma, interpolation="nearest"); plt.show()
    return returns, r_i, sigma

def StatisticCalculatorCumulativeLog(data, train_start, train_end, print_out=False, plot=False):
    prices = data.loc[train_start:train_end]
    returns = np.log(prices / prices.shift(1)).dropna()
    r_i = np.log((1 + returns).prod(axis=0)).values
    sigma = returns.cov().values

    if print_out:
        print("Cumulative Log Returns Vector (r_i):\n", r_i)
        print("\nCovariance Matrix (sigma):\n", sigma)
    if plot:
        plt.imshow(sigma, interpolation="nearest"); plt.show()
    return returns, r_i, sigma

from sklearn.covariance import LedoitWolf, OAS

def StatisticCalculatorShrinkage(data, train_start, train_end, method="LW", print_out=False, plot=False):
    returns = data.loc[train_start:train_end].pct_change().dropna()
    r_i = returns.mean().values
    est = LedoitWolf().fit(returns.values) if method.upper()=="LW" else OAS().fit(returns.values)
    sigma = est.covariance_
    if print_out:
        print("Shrinkage Returns Vector (r_i):\n", r_i)
        print("\nShrinkage Covariance Matrix (sigma):\n", sigma)
    if plot:
        plt.imshow(sigma, interpolation="nearest"); plt.show()
    return returns, r_i, sigma

def StatisticCalculatorRobust(data, train_start, train_end, print_out=False, plot=False):
    returns = data.loc[train_start:train_end].pct_change().dropna()
    r_i = returns.median().values  # robust mean (can swap to Huber)
    rho_s = returns.corr(method="spearman")
    rho_p = 2*np.sin(np.pi*rho_s/6)
    vol = returns.std()
    D = np.diag(vol.values)
    sigma = D @ rho_p.values @ D
    if print_out:
        print("Robust Returns Vector (r_i):\n", r_i)
        print("\nRobust Covariance Matrix (sigma):\n", sigma)
    if plot:
        plt.imshow(sigma, interpolation="nearest"); plt.show()
    return returns, r_i, sigma

class ReturnMethods():
    def __init__(
            self,
            data,
            train_start,
            train_end,) -> None: 
        self.data = data
        self.train_start = train_start
        self.train_end = train_end
    
    def SimpleReturn(self):
        return self.data.loc[self.train_start:self.train_end].pct_change().dropna()
    
    def LogReturn(self):
        return np.log(self.data.loc[self.train_start:self.train_end].pct_change().dropna() + 1)
        #   prices = self.data.loc[self.train_start:self.train_end]
        #   return np.log(prices / prices.shift(1)).dropna()

class ReturnEstimator():
    def __init__(
            self,
            returns,
            train_start,
            train_end,
    ) -> None:
        self.returns = returns
        self.train_start = train_start
        self.train_end = train_end

    def EWMA(self, lam: float = 0.97, use_log: bool = True):
        px = self.returns.loc[self.train_start:self.train_end]
        R = (np.log(px/px.shift(1)) if use_log else px.pct_change()).dropna()
        T = len(R)
        w = np.array([(1-lam)*(lam**k) for k in range(T)][::-1])
        w /= w.sum()
        mu = (R.values * w[:, None]).sum(axis=0)
        return pd.Series(mu, index=R.columns, name="mu_ewma")

    def CumulativeLog(self):
        px = self.returns.loc[self.train_start:self.train_end]
        R = np.log(px/px.shift(1)).dropna()
        mu = np.log((1 + R).prod(axis=0)).values  # same as log(P_T/P_0)
        return pd.Series(mu, index=R.columns, name="mu_cumlog")



class CovarianceEstimator():
    def __init__(
            self,
            train_start,
            train_end,
    ) -> None:
        self.train_start = train_start
        self.train_end = train_end

    def SampleCovariance(self, returns_df: pd.DataFrame):
        R = returns_df.loc[self.train_start:self.train_end].dropna()
        return None
    
    def EWMACovariance(self):

        return None
    
    def Shrinkage(self):
        # Ledoit-Wolf, OAS
        
        return None
    
    def Robust(self):

        return None
    
    def RandomMAtrixCleanedCovariance(self):

        return None
    
    def OptionImpliedCovariance(self):

        return None
    

def save_files():

    return None