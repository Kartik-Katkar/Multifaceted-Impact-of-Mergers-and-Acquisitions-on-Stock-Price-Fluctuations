import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Sequential, Model

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
ticker_B = 'AAPL'  # Apple Inc.
merger_announcement_date = '2020-02-20'  # Actual merger announcement date

# Convert announcement date to datetime format
announcement_date = datetime.strptime(merger_announcement_date, '%Y-%m-%d')

# Calculate dates for 60 days before and after announcement
start_date = announcement_date - timedelta(days=60)
end_date = announcement_date + timedelta(days=60)

# Fetch historical data for Company A (Morgan Stanley)
stock_data_A = fetch_stock_data(ticker_A, start_date, end_date)

# Fetch historical data for Company B (Apple Inc.)
stock_data_B = fetch_stock_data(ticker_B, start_date, end_date)

# Fetch financial ratios for Company A (Morgan Stanley)
debt_equity_A, pe_ratio_A = fetch_financial_ratios(ticker_A)

# Fetch financial ratios for Company B (Apple Inc.)
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
        'Merging company B': 'Apple Inc.',
        'Merger Announcement date': merger_announcement_date,
        'Date A': date_str,
        'Stock price A': stock_price_A,
        'Date B': date_str,
        'Stock Price B on Date B': stock_price_B,
        'Industry A': 'Finance',  # Example industry
        'Industry B': 'Technology',  # Example industry
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
real_data_df = pd.DataFrame(data)

# Save real data to CSV
real_data_df.to_csv('./Real_data.csv', index=False)

# Now integrate C-GAN generation

# Select relevant columns for training the GAN
columns = ['Stock price A', 'Stock Price B on Date B']
real_data = real_data_df[columns].dropna().values

# Scale the data to [0, 1] range
scaler = MinMaxScaler(feature_range=(0, 1))
real_data_scaled = scaler.fit_transform(real_data)

# Define the generator model with modified output for continuous trends
def build_generator(latent_dim, output_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(output_dim, activation='linear'))  # Output layer modified for continuous trends
    model.add(Reshape((output_dim,)))
    return model

# Define the discriminator model
def build_discriminator(input_dim):
    model = Sequential()
    model.add(Flatten(input_shape=(input_dim,)))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Define the GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = Input(shape=(latent_dim,))
    generated_data = generator(gan_input)
    gan_output = discriminator(generated_data)
    gan = Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan

# Set dimensions
latent_dim = 100
output_dim = real_data_scaled.shape[1]

# Build and compile models
generator = build_generator(latent_dim, output_dim)
discriminator = build_discriminator(output_dim)
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
gan = build_gan(generator, discriminator)

# Function to train the GAN with continuous trends
def train_gan(gan, generator, discriminator, data, epochs, batch_size):
    half_batch = int(batch_size / 2)

    for epoch in range(epochs):
        # Train the discriminator with real data
        idx = np.random.randint(0, data.shape[0], half_batch)
        real_samples = data[idx]
        real_labels = np.ones((half_batch, 1))

        d_loss_real = discriminator.train_on_batch(real_samples, real_labels)

        # Train the discriminator with fake data
        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        fake_samples = generator.predict(noise)
        fake_labels = np.zeros((half_batch, 1))

        d_loss_fake = discriminator.train_on_batch(fake_samples, fake_labels)

        # Train the generator to generate data with continuous trends
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_labels = np.ones((batch_size, 1))
        generated_samples = generator.predict(noise)
        
        # Penalize the generator if the generated data does not follow trends
        g_loss = gan.train_on_batch(noise, valid_labels)

        # Print the progress
        if epoch % 100 == 0:
            print(f"{epoch} [D loss: {0.5 * np.add(d_loss_real, d_loss_fake)}] [G loss: {g_loss}]")

# Train the GAN with continuous trends
epochs = 500
batch_size = 64
train_gan(gan, generator, discriminator, real_data_scaled, epochs, batch_size)

# Generate synthetic data
num_samples = real_data_scaled.shape[0]
noise = np.random.normal(0, 1, (num_samples, latent_dim))
synthetic_data_scaled = generator.predict(noise)

# Inverse transform to get the actual scale
synthetic_data = scaler.inverse_transform(synthetic_data_scaled)

# Create a DataFrame for the synthetic data
synthetic_data_df = pd.DataFrame(synthetic_data, columns=columns)

# Merge with original DataFrame for completeness
result_df = real_data_df.copy()
result_df[['Synthetic Stock price A', 'Synthetic Stock Price B']] = synthetic_data_df

# Save synthetic data to CSV
result_df.to_csv('./cganneddata.csv', index=False)

print("Real and synthetic data generated and saved successfully.")

import pandas as pd
import matplotlib.pyplot as plt

# Load the generated synthetic data and C-GAN data
synthetic_data_df = pd.read_csv('./cganneddata.csv')

# Plotting synthetic stock prices
plt.figure(figsize=(10, 6))
plt.plot(synthetic_data_df['Synthetic Stock price A'], label='Synthetic Stock Price A')
plt.plot(synthetic_data_df['Synthetic Stock Price B'], label='Synthetic Stock Price B')
plt.title('Synthetic Stock Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plotting C-GAN generated stock prices
plt.figure(figsize=(10, 6))
plt.plot(synthetic_data_df['Stock price A'], label='Stock Price A')
plt.plot(synthetic_data_df['Stock Price B on Date B'], label='Stock Price B')
plt.title('C-GAN Generated Stock Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()