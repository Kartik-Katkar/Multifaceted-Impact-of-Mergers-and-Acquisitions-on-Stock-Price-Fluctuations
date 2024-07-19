import os
import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from datetime import datetime

# Define ticker list
ticker2_list = [
    "GOOG", "AMZN", "MSFT", "TSLA", "META", "JNJ", "UNH", "XOM", "NVDA", "BAC",
    "DIS", "T", "DOW", "CVS", "CHTR", "IBM", "LIN", "LLY", "ABBV", "WFC",
    "HD", "PG", "MRK", "NEE", "NKE", "NFLX", "DUK", "MO", "HON",
    "KO", "PEP", "MCD", "CSCO", "AAPL", "PM", "INTC", "WMT", "AXP", "COST",
    "BA", "JPM", "VZ", "MAR", "AVGO", "CMCSA", "BLK"
]

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

# Function to optimize portfolio
def optimize_portfolio(tickers, start_date, end_date):
    # Download historical stock price data
    data = fetch_stock_data(tickers, start_date, end_date)
    if data is None:
        return None, None, None, None, None

    # Compute daily returns
    returns = data.pct_change().dropna()

    # Compute the correlation matrix
    correlation_matrix = returns.corr()

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

# Function to plot stock predictions
def plot_stock_predictions(tickers, start_date, end_date):
    for ticker in tickers:
        file_path = os.path.join("Dataset", f"{ticker}_with_predictedvalues.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            plt.figure(figsize=(10, 6))
            plt.plot(df['Date'], df['Original'], label='Original')
            plt.plot(df['Date'], df['Predicted'], label='Predicted')
            plt.title(f"{ticker} Stock Predictions")
            plt.xlabel("Date")
            plt.ylabel("Stock Price")
            plt.legend()
            plt.show()
        else:
            messagebox.showerror("File Error", f"Could not find {ticker} data file at {file_path}.")

# Function to handle portfolio optimization button click
def optimize_portfolio_clicked():
    ticker1 = ticker1_entry.get().upper()
    start_date = start_date_entry.get()
    end_date = end_date_entry.get()

    try:
        # Validate ticker1
        # Validate if ticker1 is in ticker2_list
        if ticker1 not in ticker2_list:
            raise ValueError(f"Ticker {ticker1} is not in the predefined ticker list.")
        else:
            # Remove ticker1 from ticker2_list if it's present
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

            # Include ticker1 in the tickers list
            tickers = [ticker1] + top_high_corr_tickers
            # Optimize the portfolio
            optimized_weights, cumulative_returns, annualized_return, annualized_volatility, sharpe_ratio = optimize_portfolio(tickers, start_date, end_date)
            # Output optimization results to GUI
            if optimized_weights is not None:
                optimized_weights_str = [f'{weight:.2f}' for weight in optimized_weights]
                optimized_weights_var.set(optimized_weights_str)
                annualized_return_var.set(f"{annualized_return:.2f}")
                annualized_volatility_var.set(f"{annualized_volatility:.2f}")
                sharpe_ratio_var.set(f"{sharpe_ratio:.2f}")
                
                # Check if ticker1 is in the specified list and update the label
                if ticker1 in ["UBS", "GOOG", "MRK", "DIS"]:
                    made_using_label.config(text="made using merger model")
                else:
                    made_using_label.config(text="")

            else:
                messagebox.showerror("Optimization Error", "Portfolio optimization failed. Please check the data and try again.")

                # Plot stock predictions for all tickers
        plot_stock_predictions(tickers, start_date, end_date)

    except ValueError as ve:
        messagebox.showerror("Value Error", str(ve))

    except Exception as e:
        messagebox.showerror("Error", str(e))

# GUI setup
root = tk.Tk()
root.title("Portfolio Optimization and Stock Predictions")

# Main frame
main_frame = ttk.Frame(root, padding="20")
main_frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Ticker1 input
ttk.Label(main_frame, text="Enter the main ticker symbol:").grid(column=0, row=0, sticky=tk.W)
ticker1_entry = ttk.Entry(main_frame, width=10)
ticker1_entry.grid(column=1, row=0, sticky=tk.W)

# Start date input
ttk.Label(main_frame, text="Start Date (YYYY-MM-DD):").grid(column=0, row=1, sticky=tk.W)
start_date_entry = ttk.Entry(main_frame, width=10)
start_date_entry.grid(column=1, row=1, sticky=tk.W)
start_date_entry.insert(0, "2020-01-01")  # Default start date

# End date input
ttk.Label(main_frame, text="End Date (YYYY-MM-DD):").grid(column=0, row=2, sticky=tk.W)
end_date_entry = ttk.Entry(main_frame, width=10)
end_date_entry.grid(column=1, row=2, sticky=tk.W)
end_date_entry.insert(0, "2024-07-17")  # Default end date

# Buttons
optimize_portfolio_button = ttk.Button(main_frame, text="Optimize Portfolio", command=optimize_portfolio_clicked)
optimize_portfolio_button.grid(column=0, row=3, sticky=tk.W)

# Output labels
optimized_weights_label = ttk.Label(main_frame, text="Optimized Weights:")
optimized_weights_label.grid(column=0, row=4, sticky=tk.W)
optimized_weights_var = tk.StringVar()
ttk.Label(main_frame, textvariable=optimized_weights_var).grid(column=1, row=4, sticky=tk.W)

annualized_return_label = ttk.Label(main_frame, text="Annualized Return:")
annualized_return_label.grid(column=0, row=5, sticky=tk.W)
annualized_return_var = tk.StringVar()
ttk.Label(main_frame, textvariable=annualized_return_var).grid(column=1, row=5, sticky=tk.W)

annualized_volatility_label = ttk.Label(main_frame, text="Annualized Volatility:")
annualized_volatility_label.grid(column=0, row=6, sticky=tk.W)
annualized_volatility_var = tk.StringVar()
ttk.Label(main_frame, textvariable=annualized_volatility_var).grid(column=1, row=6, sticky=tk.W)

sharpe_ratio_label = ttk.Label(main_frame, text="Sharpe Ratio:")
sharpe_ratio_label.grid(column=0, row=7, sticky=tk.W)
sharpe_ratio_var = tk.StringVar()
ttk.Label(main_frame, textvariable=sharpe_ratio_var).grid(column=1, row=7, sticky=tk.W)

# Additional label for "made using merger model"
made_using_label = ttk.Label(main_frame, text="", font=("Arial", 8, "italic"))
made_using_label.grid(column=1, row=8, sticky=tk.E)

# Adjust layout
for child in main_frame.winfo_children():
    child.grid_configure(padx=5, pady=5)

# Start GUI
root.mainloop()