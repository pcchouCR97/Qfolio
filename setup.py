"""
Qfolio - Quantum Portfolio Optimization Framework
===================================================

A Python package for quantum-enhanced portfolio optimization using QAOA, VQE,
and classical solvers with AMSP covariance estimation and CVaR risk management.

Author: Po-Chih Chou
License: MIT
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_file(filename):
    with open(os.path.join(os.path.dirname(__file__), filename), encoding='utf-8') as f:
        return f.read()

# Read requirements
def read_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='qfolio',
    version='0.1.0',
    author='Po-Chih Chou',
    author_email='pcchouCR97@gmail.com',
    description='Quantum-enhanced portfolio optimization with AMSP and CVaR',
    long_description=read_file('README.md') if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    url='https://github.com/pcchouCR97/Qfolio', 

    # Automatically find all packages
    packages=find_packages(exclude=['tests', 'Tutorial', 'VOO_testbed_data', 'quantstats_reports']),

    # Include non-Python files (data, configs, etc.)
    include_package_data=True,
    package_data={
        'qfolio': ['data/*.csv', 'config/*.yaml'],
    },

    # Python version requirement
    python_requires='>=3.11',

    # Dependencies
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'scipy>=1.7.0',
        'scikit-learn>=1.0.0',
        'yfinance>=0.2.0',
        'pandas-market-calendars>=4.0.0',
        'quantstats>=0.0.59',
        'qiskit>=0.45.0',
        'qiskit-optimization>=0.5.0',
        'qiskit-algorithms>=0.2.0',
        'qiskit-aer>=0.12.0',
        'cvxpy>=1.3.0',
        'pyscipopt>=4.3.0',
    ],

    # Optional dependencies (install with: pip install qfolio[dev])
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'mypy>=0.950',
            'jupyter>=1.0.0',
            'jupyterlab>=3.4.0',
        ],
        'docs': [
            'sphinx>=4.5.0',
            'sphinx-rtd-theme>=1.0.0',
            'nbsphinx>=0.8.0',
        ],
    },

    # Entry points (CLI commands)
    entry_points={
        'console_scripts': [
            'qfolio-compare=qfolio.cli:compare_solvers',
            'qfolio-backtest=qfolio.cli:run_backtest',
        ],
    },

    # Classifiers for PyPI
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Office/Business :: Financial :: Investment',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],

    # Keywords for discoverability
    keywords='quantum computing portfolio optimization QAOA VQE finance quant',

    # Project URLs
    project_urls={
        'Documentation': 'https://qfolio.readthedocs.io/',
        'Source': 'https://github.com/pcchouCR97/Qfolio',
        'Bug Reports': 'https://github.com/pcchouCR97/Qfolio/issues',
    },
)
