import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import requests

# Function to fetch historical data using yfinance
def fetch_stock_data(ticker, start, end):
    stock_data = yf.download(ticker, start=start, end=end)
    return stock_data

# Function to fetch financial ratios (Debt to Equity and Price Earnings Ratio) from financialmodelingprep API
def fetch_financial_ratios(ticker):
    url = f'https://financialmodelingprep.com/api/v3/ratios/{ticker}?apikey=yT72Tp1YrJehGBySNBrAgwc5WZ0ZnSif'
    response = requests.get(url)
    if response.status_code == 200:
        ratios = response.json()
        if ratios:
            return ratios[0].get('debtEquityRatio'), ratios[0].get('priceEarningsRatio')
    return None, None

# Define tickers and announcement date
ticker_A = 'MS'  # Morgan Stanley
ticker_B = 'ETFC'  # E*TRADE
merger_announcement_date = '2020-02-20'  # Actual merger announcement date

# Convert announcement date to datetime format
announcement_date = datetime.strptime(merger_announcement_date, '%Y-%m-%d')

# Calculate dates for 60 days before and after announcement
start_date = announcement_date - timedelta(days=60)
end_date = announcement_date + timedelta(days=60)

# Fetch historical data for Company A (Morgan Stanley)
stock_data_A = fetch_stock_data(ticker_A, start_date, end_date)

# Fetch historical data for Company B (E*TRADE)
stock_data_B = fetch_stock_data(ticker_B, start_date, end_date)

# Fetch financial ratios for Company A (Morgan Stanley)
debt_equity_A, pe_ratio_A = fetch_financial_ratios(ticker_A)

# Fetch financial ratios for Company B (E*TRADE)
debt_equity_B, pe_ratio_B = fetch_financial_ratios(ticker_B)

# Prepare data for the specific dates
data = []

for date in pd.date_range(start=start_date, end=end_date):
    date_str = date.strftime('%Y-%m-%d')
    
    # Get data for Company A
    if date_str in stock_data_A.index:
        row_A = stock_data_A.loc[date_str]
        stock_price_A = row_A['Close']
    else:
        stock_price_A = None

    # Get data for Company B
    if date_str in stock_data_B.index:
        row_B = stock_data_B.loc[date_str]
        stock_price_B = row_B['Close']
    else:
        stock_price_B = None

    data.append({
        'Merging company A': 'Morgan Stanley',
        'Merging company B': 'E*TRADE',
        'Merger Announcement date': merger_announcement_date,
        'Date A': date_str,
        'Stock price A': stock_price_A,
        'Date B': date_str,
        'Stock Price B on Date B': stock_price_B,
        'Industry A': 'Finance',  # Example industry
        'Industry B': 'Finance',  # Example industry
        'percentage ownership A': None,  # Placeholder for data not available
        'percentage ownership B': None,  # Placeholder for data not available
        'Debt to Equity A': debt_equity_A,
        'Debt to Equity B': debt_equity_B,
        'Paid amount by A': None,  # Placeholder for data not available
        'Paid amount by B': None,  # Placeholder for data not available
        'Price Earnings Ratio A': pe_ratio_A,
        'Price Earnings Ratio B': pe_ratio_B
    })

# Convert the prepared data into a DataFrame
df = pd.DataFrame(data)

# Append to an existing CSV or create a new one
df.to_csv('./Synthesized data/Real data.csv', mode='a', header=False, index=False)

print("Data appended successfully.")
