import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load CSV data for multiple companies
def load_combined_data(files):
    combined_data = pd.DataFrame()
    for file in files:
        df = pd.read_csv(file)
        combined_data = pd.concat([combined_data, df])
    return combined_data

# Train model on combined data
def train_combined_model(merged_dates, data_files):
    # Load combined data
    combined_data = load_combined_data(data_files)
    
    # Example: Feature engineering and preparation
    # Assuming 'adjClose' is the target variable
    X = combined_data[['date', 'close', 'high', 'low', 'open', 'volume', 'adjClose', 'adjHigh', 'adjLow', 'adjOpen', 'adjVolume']]
    y = combined_data['adjClose']
    
    # Example: Train a model
    model = RandomForestRegressor()
    model.fit(X, y)
    
    # Save the model using pickle
    model_file = "merger_model.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Combined model saved as {model_file}")

# Example usage with multiple company data files and merger dates
data_files = ['./BLK_prices_2023-12-31_2024-03-31.csv', './DIS_prices_2005-11-30_2006-03-31.csv', './MRK_prices_2009-01-01_2009-05-31.csv', './UBS_prices_2023-02-01_2024-06-30.csv']
merged_dates = ['2024-01-12', '2006-01-24', '2009-11-03', '2023-03-19']  # Example merger dates for each company

train_combined_model(merged_dates, data_files)
