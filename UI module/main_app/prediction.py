import yfinance as yf
import pandas as pd
from scipy.optimize import minimize
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def calculate_correlations(ticker1, ticker2_list,risk, start_date="2020-01-01", end_date="2024-07-17"):
    try:
        # Remove ticker1 from ticker2_list if it's present
        if ticker1 in ticker2_list:
            ticker2_list.remove(ticker1)

        # Download data using yfinance
        data = yf.download([ticker1] + ticker2_list, start=start_date, end=end_date)['Close']

        # Check if data is retrieved successfully
        if data.empty:
            raise ValueError("No data downloaded. Check if ticker symbols or date range are correct.")

        # Calculate the correlation coefficients
        correlations = data.corr(method="pearson")

        # Filter correlations greater than 0.8 and less than -0.8
        high_correlation = []
        low_correlation = []

        risk_lvl = 0

        if risk == "low_risk":
            risk = 0.85
        elif risk == "medium_risk":
            risk = 0.5
        elif risk == "high_risk":
            risk = 0.3


        for ticker2 in ticker2_list:
            correlation_value = correlations.loc[ticker1, ticker2]
            if correlation_value > risk:
                high_correlation.append((ticker2, correlation_value))
            elif correlation_value < -risk:
                low_correlation.append((ticker2, correlation_value))

        # Sort by correlation value
        high_correlation.sort(key=lambda x: x[1], reverse=True)
        low_correlation.sort(key=lambda x: x[1])

                # Save top 4 and bottom 4 tickers
        top_high_corr_tickers = [ticker[0] for ticker in high_correlation[:4]]
        bottom_low_corr_tickers = [ticker[0] for ticker in low_correlation[:4]]

        # Return results
        return top_high_corr_tickers, bottom_low_corr_tickers

    except ValueError as ve:
        print("ValueError:", ve)
        return None, None

    except KeyError as ke:
        print("KeyError:", ke)
        return None, None

    except Exception as e:
        print("Error:", e)
        return None, None

# Function to fetch historical stock data
def fetch_stock_data(tickers, start_date, end_date):
    try:
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def portfolio_performance(weights, returns):
    return np.dot(weights, returns.T)

def optimize_portfolio(tickers, start_date, end_date):
    # Download historical stock price data
    data = fetch_stock_data(tickers, start_date, end_date)
    if data is None:
        return None, None, None, None, None

    # Compute daily returns
    returns = data.pct_change().dropna()

    # Compute the correlation matrix
    correlation_matrix = returns.corr()

    # # Visualize the correlation matrix
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    # plt.title('Stock Return Correlation Matrix')
    # plt.show()

    # Define initial guess for weights
    initial_weights = np.array([1 / len(tickers)] * len(tickers))

    # Define mean returns and covariance matrix
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # Define constraints and bounds for optimization
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(tickers)))

    # Optimize portfolio for minimum volatility
    try:
        result = minimize(portfolio_volatility, initial_weights, args=(mean_returns, cov_matrix),
                          method='SLSQP', bounds=bounds, constraints=constraints)

        # Extract optimized weights
        optimized_weights = result.x

        # Calculate portfolio returns
        portfolio_returns = portfolio_performance(optimized_weights, returns)

        # Calculate performance metrics
        cumulative_returns = (1 + portfolio_returns).cumprod()
        annualized_return = np.mean(portfolio_returns) * 252
        annualized_volatility = np.std(portfolio_returns) * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_volatility

        return optimized_weights, cumulative_returns, annualized_return, annualized_volatility, sharpe_ratio

    except Exception as e:
        print(f"Error optimizing portfolio: {e}")
        return None, None, None, None, None

# Define tickers and date range