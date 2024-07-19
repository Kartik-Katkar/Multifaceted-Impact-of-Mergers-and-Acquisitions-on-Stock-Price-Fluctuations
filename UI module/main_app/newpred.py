import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import streamlit as st

# Function to fetch historical stock data
def fetch_stock_data(tickers, start_date, end_date):
    try:
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# Function to calculate portfolio volatility
def portfolio_volatility(weights, mean_returns, cov_matrix):
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_volatility = np.sqrt(portfolio_variance)
    return portfolio_volatility

# Function to calculate portfolio returns
def portfolio_performance(weights, returns):
    return np.dot(weights, returns.T)

# Main function for portfolio optimization
def optimize_portfolio(tickers, start_date, end_date):
    # Download historical stock price data
    data = fetch_stock_data(tickers, start_date, end_date)
    if data is None:
        return None, None, None, None, None

    # Compute daily returns
    returns = data.pct_change().dropna()

    # Compute the correlation matrix
    correlation_matrix = returns.corr()

    # Visualize the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Stock Return Correlation Matrix')
    plt.show()

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

ticker1 = "AAPL"
ticker2_list = [
    "GOOG", "AMZN", "MSFT", "TSLA", "META", "JNJ", "UNH", "XOM", "NVDA", "BAC",
    "DIS", "T", "DOW", "CVS", "CHTR", "IBM", "LIN", "LLY", "ABBV", "WFC",
    "HD", "PG", "MRK", "NEE", "NKE", "NFLX", "DUK", "MO", "HON",
    "KO", "PEP", "MCD", "CSCO", "AAPL", "PM", "INTC", "WMT", "AXP", "COST",
    "BA", "JPM", "VZ", "MAR", "AVGO", "CMCSA"
]

start_date = "2020-01-01"
end_date = "2024-07-17"

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

    for ticker2 in ticker2_list:
        correlation_value = correlations.loc[ticker1, ticker2]
        if correlation_value > 0.8:
            high_correlation.append((ticker2, correlation_value))
        elif correlation_value < -0.8:
            low_correlation.append((ticker2, correlation_value))

    # Sort by correlation value
    high_correlation.sort(key=lambda x: x[1], reverse=True)
    low_correlation.sort(key=lambda x: x[1])

    # Save top 4 and bottom 4 tickers
    top_high_corr_tickers = [ticker[0] for ticker in high_correlation[:4]]
    bottom_low_corr_tickers = [ticker[0] for ticker in low_correlation[:4]]

    # Print results
    print("Top 4 tickers correlated with", ticker1, ">", top_high_corr_tickers)
    print("Bottom 4 tickers correlated with", ticker1, "<", bottom_low_corr_tickers)

except ValueError as ve:
    print("ValueError:", ve)

except KeyError as ke:
    print("KeyError:", ke)

except Exception as e:
    print("Error:", e)

tickers = top_high_corr_tickers  # Removed 'FB' due to data fetch error
start_date = "2020-01-01"
end_date = "2024-01-01"

# Optimize the portfolio
optimized_weights, cumulative_returns, annualized_return, annualized_volatility, sharpe_ratio = optimize_portfolio(tickers, start_date, end_date)

# Print results
if optimized_weights is not None:
    print("Optimized Weights:", optimized_weights)
    print(f"Annualized Return: {annualized_return:.2f}")
    print(f"Annualized Volatility: {annualized_volatility:.2f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
else:
    print("Portfolio optimization failed.")

# Streamlit Dashboard
if optimized_weights is not None:
    st.title("Portfolio Optimization Dashboard")
    st.header("Optimized Portfolio Weights")
    weights_df = pd.DataFrame(optimized_weights, index=tickers, columns=['Weights'])

# Display bar chart with weights
    st.bar_chart(weights_df)

    st.header("Cumulative Returns")
    st.line_chart(cumulative_returns)

    st.header("Performance Metrics")
    st.write(f"Annualized Return: {annualized_return:.2f}")
    st.write(f"Annualized Volatility: {annualized_volatility:.2f}")
    st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")
else:
    st.write("Portfolio optimization failed. Please check the data and try again.")