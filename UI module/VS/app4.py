from flask import Flask, render_template
import pandas as pd
import plotly.express as px
import plotly.io as pio
import os
import yfinance as yf
import numpy as np
from scipy.optimize import minimize

app = Flask(__name__)

# Function to load CSV data for each ticker
def load_data(ticker):
    file_path = f"Dataset/{ticker}_with_predictedvalues.csv"
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        print(f"File not found: {file_path}")
        return None

# Function to create Plotly plot from data
def create_plot(data, title):
    fig = px.line(data, x='Date', y=['Original', 'Predicted'], title=title)
    return pio.to_html(fig, full_html=False)

# Function to fetch stock data from Yahoo Finance
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
    data = fetch_stock_data(tickers, start_date, end_date)
    if data is None:
        return None, None, None, None, None

    returns = data.pct_change().dropna()
    correlation_matrix = returns.corr()

    initial_weights = np.array([1 / len(tickers)] * len(tickers))
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(tickers)))

    try:
        result = minimize(portfolio_volatility, initial_weights, args=(mean_returns, cov_matrix),
                          method='SLSQP', bounds=bounds, constraints=constraints)

        optimized_weights = result.x
        portfolio_returns = portfolio_performance(optimized_weights, returns)

        cumulative_returns = (1 + portfolio_returns).cumprod()[-1]  # Only the last value
        annualized_return = np.mean(portfolio_returns) * 252
        annualized_volatility = np.std(portfolio_returns) * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_volatility

        return optimized_weights.tolist(), cumulative_returns, annualized_return, annualized_volatility, sharpe_ratio

    except Exception as e:
        print(f"Error optimizing portfolio: {e}")
        return None, None, None, None, None

def sex(ticker1):
    ticker2_list = [
        "GOOG", "AMZN", "MSFT", "TSLA", "META", "JNJ", "UNH", "XOM", "NVDA", "BAC",
        "DIS", "T", "DOW", "CVS", "CHTR", "IBM", "LIN", "LLY", "ABBV", "WFC",
        "HD", "PG", "MRK", "NEE", "NKE", "NFLX", "DUK", "MO", "HON",
        "KO", "PEP", "MCD", "CSCO", "AAPL", "PM", "INTC", "WMT", "AXP", "COST",
        "BA", "JPM", "VZ", "MAR", "AVGO", "CMCSA","BLK","UBS"
    ]

    start_date = "2020-01-01"
    end_date = "2024-07-17"

    try:
        if ticker1 in ticker2_list:
            ticker2_list.remove(ticker1)

        data = yf.download([ticker1] + ticker2_list, start=start_date, end=end_date)['Close']

        if data.empty:
            raise ValueError("No data downloaded. Check ticker symbols or date range.")

        correlations = data.corr(method="pearson")

        high_correlation = []
        low_correlation = []

        for ticker2 in ticker2_list:
            correlation_value = correlations.loc[ticker1, ticker2]
            if correlation_value > 0.8:
                high_correlation.append((ticker2, correlation_value))
            elif correlation_value < -0.8:
                low_correlation.append((ticker2, correlation_value))

        high_correlation.sort(key=lambda x: x[1], reverse=True)
        low_correlation.sort(key=lambda x: x[1])

        top_high_corr_tickers = [ticker[0] for ticker in high_correlation[:4]]
        bottom_low_corr_tickers = [ticker[0] for ticker in low_correlation[:4]]

    except ValueError as ve:
        print("ValueError:", ve)
    except KeyError as ke:
        print("KeyError:", ke)
    except Exception as e:
        print("Error:", e)

    tickers =  top_high_corr_tickers
    start_date = "2020-01-01"
    end_date = "2024-01-01"

    optimized_weights, cumulative_returns, annualized_return, annualized_volatility, sharpe_ratio = optimize_portfolio(tickers, start_date, end_date)

    return top_high_corr_tickers , optimized_weights ,cumulative_returns, annualized_return, annualized_volatility, sharpe_ratio

def create_pie_chart(weights, tickers):
    data = {
        'Ticker': tickers,
        'Weight': weights
    }
    df = pd.DataFrame(data)
    fig = px.pie(df, values='Weight', names='Ticker', title='Optimized Portfolio Weights')
    return pio.to_html(fig, full_html=False)

@app.route('/')
def index():
        
    #open and read the file after the appending:
    f = open("../main_app/varex.txt", "r")
    company = str(f.read())
    f.close()
    # os.remove("../main_app/varex.txt")
    tickers , optimized_weights , cumulative_returns, annualized_return, annualized_volatility, sharpe_ratio = sex(company)

    graphs = []
    data1 = load_data(company)
    graph = create_plot(data1, f'{company} Stock Predictions')
    graphs.append(graph)
    for ticker in tickers:
        data = load_data(ticker)
        if data is not None:
            graph = create_plot(data, f'{ticker} Stock Predictions')
            graphs.append(graph)
    
    # Multiply values by 100 and format to 2 decimal places
    optimized_weights = [round(float(w) * 100, 2) for w in optimized_weights]
    cumulative_returns = round(float(cumulative_returns) * 100, 2)
    annualized_return = round(float(annualized_return) * 100, 2)
    annualized_volatility = round(float(annualized_volatility) * 100, 2)
    sharpe_ratio = round(float(sharpe_ratio), 2)

    pie_chart = create_pie_chart(optimized_weights, tickers)

    return render_template('index.html', graphs=graphs,
                           optimized_weights=optimized_weights,
                           cumulative_returns=cumulative_returns,
                           annualized_return=annualized_return,
                           annualized_volatility=annualized_volatility,
                           sharpe_ratio=sharpe_ratio,
                           pie_chart=pie_chart,company=company)

if __name__ == '__main__':
    app.run(debug=True, port=5004)
