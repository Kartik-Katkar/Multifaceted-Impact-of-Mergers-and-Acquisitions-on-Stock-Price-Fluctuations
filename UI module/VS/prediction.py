import yfinance as yf
import pandas as pd

def calculate_correlations(ticker1, company_type, start_date="2020-01-01", end_date="2024-07-17"):
    try:
        ticker2_list = [
        "GOOG", "AMZN", "MSFT", "TSLA", "META", "JNJ", "UNH", "XOM", "NVDA", "BAC",
        "DIS", "T", "DOW", "CVS", "CHTR", "IBM", "LIN", "LLY", "ABBV", "WFC",
        "HD", "PG", "MRK", "NEE", "NKE", "NFLX", "DUK", "MO", "HON",
        "KO", "PEP", "MCD", "CSCO", "AAPL", "PM", "INTC", "WMT", "AXP", "COST",
        "BA", "JPM", "VZ", "MAR", "AVGO", "CMCSA"]


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
        high_correlation = set()
        low_correlation = set()

        for ticker2 in ticker2_list:
            correlation_value = correlations.loc[ticker1, ticker2]
            if correlation_value > 0.8:
                high_correlation.add((ticker2, correlation_value))
            elif correlation_value < -0.8:
                low_correlation.add((ticker2, correlation_value))

        # Return results
        return list(high_correlation), list(low_correlation)

    except ValueError as ve:
        print("ValueError:", ve)
        return None, None

    except KeyError as ke:
        print("KeyError:", ke)
        return None, None

    except Exception as e:
        print("Error:", e)
        return None, None

# Example usage:
# if __name__ == "__main__":
#     ticker1 = "AAPL"
#     ticker2_list = [
#         "GOOG", "AMZN", "MSFT", "TSLA", "META", "JNJ", "UNH", "XOM", "NVDA", "BAC",
#         "DIS", "T", "DOW", "CVS", "CHTR", "IBM", "LIN", "LLY", "ABBV", "WFC",
#         "HD", "PG", "MRK", "NEE", "NKE", "NFLX", "DUK", "MO", "HON",
#         "KO", "PEP", "MCD", "CSCO", "AAPL", "PM", "INTC", "WMT", "AXP", "COST",
#         "BA", "JPM", "VZ", "MAR", "AVGO", "CMCSA"
#     ]

#     high_correlation, low_correlation = calculate_correlations(ticker1, ticker2_list)
#     if high_correlation:
#         print(f"Tickers correlated with {ticker1} > 0.8:", high_correlation)
#     if low_correlation:
#         print(f"Tickers correlated with {ticker1} < -0.8:", low_correlation)
