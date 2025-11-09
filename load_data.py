import pandas as pd
import yfinance as yf
from data import assets

def fetch_and_store_data(file_path="asset_prices.csv"):
    data = yf.download(assets, start="2015-01-01", end="2024-01-01", interval="1d")['Adj Close']

    monthly_prices = data.resample('M').last()
    monthly_returns = monthly_prices.pct_change().dropna()

    combined_data = monthly_prices.copy()
    combined_data['Year'] = combined_data.index.year
    combined_data['Month'] = combined_data.index.month_name()
    for asset in assets:
        combined_data[f"{asset}_Return"] = monthly_returns[asset]

    combined_data.reset_index(inplace=True)
    combined_data.rename(columns={'Date': 'End_of_Month'}, inplace=True)
    combined_data = combined_data[['Year', 'Month'] + [col for col in combined_data.columns if col not in ['Year', 'Month', 'End_of_Month']]]

    combined_data.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

if __name__ == "__main__":
    fetch_and_store_data()
