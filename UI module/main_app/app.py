from flask import Flask, render_template, redirect, request, url_for
from prediction import calculate_correlations,optimize_portfolio  # Ensure you have this import
import pandas as pd
import plotly.graph_objs as go
from os import listdir
from os.path import isfile, join

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/app1')
def app1():
    return redirect('http://localhost:5001')

@app.route('/app2')
def app2():
    return redirect('http://localhost:5002')

@app.route('/app3')
def app3():
    return redirect('http://localhost:5003')

@app.route('/app4')
def app4():
    return redirect('http://localhost:5004')

@app.route('/company', methods=['POST'])
def company():
    company_name = request.form.get('company_name')
    #open and read the file after the appending:
    f = open("varex.txt", "w")
    print(f.write(company_name))
    f.close()
    company_type = request.form.get('company_type')
    
    # You can process company_type as needed
    
    # Get correlations
    ticker2_list = [
        "GOOG", "AMZN", "MSFT", "TSLA", "META", "JNJ", "UNH", "XOM", "NVDA", "BAC",
        "DIS", "T", "DOW", "CVS", "CHTR", "IBM", "LIN", "LLY", "ABBV", "WFC",
        "HD", "PG", "MRK", "NEE", "NKE", "NFLX", "DUK", "MO", "HON",
        "KO", "PEP", "MCD", "CSCO", "PM", "INTC", "WMT", "AXP", "COST",
        "BA", "JPM", "VZ", "MAR", "AVGO", "CMCSA"
    ]

    # high_correlation, low_correlation = calculate_correlations(company_name, ticker2_list, company_type)

    # tickers = high_correlation  # Removed 'FB' due to data fetch error
    # start_date = "2020-01-01"
    # end_date = "2024-01-01"

    # # Optimize the portfolio
    # optimized_weights, cumulative_returns, annualized_return, annualized_volatility, sharpe_ratio = optimize_portfolio(tickers, start_date, end_date)

    # print(optimized_weights, cumulative_returns, annualized_return, annualized_volatility, sharpe_ratio)
    
    # return render_template('results.html', high_correlation=high_correlation, low_correlation=low_correlation)
    return redirect('http://localhost:5004')

if __name__ == '__main__':
    app.run(port=5000)
