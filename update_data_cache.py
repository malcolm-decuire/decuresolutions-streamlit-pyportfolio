# update_data_cache.py

import csv
from datetime import datetime
import numpy as np
from dateutil.relativedelta import relativedelta
import pandas as pd
import pandas_datareader as pdr
import yfinance as yf
from pypfopt import expected_returns, risk_models

# Function to fetch asset prices
def fetch_asset_prices(asset_list=None, start=None, end=None):
    if not asset_list:
        with open('inputs/assets.csv', newline='') as f:
            reader = csv.reader(f)
            asset_list = [row[0] for row in reader]

    # Default time range
    start = start or (datetime.now() - relativedelta(years=10))
    end = end or datetime.now()

    # Fetch price data
    asset_prices = yf.download(asset_list, start=start, end=end, progress=False)
    asset_prices = asset_prices.filter(like='Adj Close')  # Filter columns
    asset_prices.columns = asset_prices.columns.get_level_values(1)  # Flatten column levels

    # Drop assets with insufficient data
    valid_cols = asset_prices.isin([' ', 'NULL', np.nan]).mean() < 0.8
    asset_prices = asset_prices.loc[:, valid_cols]

    return asset_prices

# Function to fetch risk-free rate
def fetch_risk_free_rate(start, end):
    risk_free_rate = pdr.DataReader("IRLTLT01USM156N", "fred", start, end)
    return risk_free_rate.iloc[-1].item() / 100

# Function to compute expected returns and covariance matrix
def compute_portfolio_metrics(prices, risk_free_rate):
    e_returns = expected_returns.capm_return(prices, risk_free_rate=risk_free_rate)
    cov_mat = risk_models.exp_cov(prices)
    return e_returns, cov_mat

# Main function to fetch all data and metrics
def get_data(asset_list=None):
    start = datetime.now() - relativedelta(years=10)
    end = datetime.now()

    prices = fetch_asset_prices(asset_list, start, end)
    risk_free_rate = fetch_risk_free_rate(start, end)
    e_returns, cov_mat = compute_portfolio_metrics(prices, risk_free_rate)

    return e_returns, cov_mat, risk_free_rate

# Save data to disk for caching
def save_data_to_disk(e_returns, cov_mat, risk_free_rate):
    e_returns.to_csv('inputs/e_returns.csv')
    cov_mat.to_csv('inputs/cov_mat.csv')
    with open('inputs/risk_free_rate.txt', 'w') as f:
        f.write('%f' % risk_free_rate)

# If run as a script, fetch and save data
if __name__ == "__main__":
    e_returns, cov_mat, risk_free_rate = get_data()
    save_data_to_disk(e_returns, cov_mat, risk_free_rate)
