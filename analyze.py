import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data import assets, sectors, total_capital

graphs_dir = "graphs"
os.makedirs(graphs_dir, exist_ok=True)

results_file = "model_results.csv"
data = pd.read_csv(results_file)

shares_allocation_file = "shares_allocation.csv"
shares_data = pd.read_csv(shares_allocation_file)

asset_prices_file = "asset_prices.csv"
asset_prices_data = pd.read_csv(asset_prices_file)

def analyze_results():
    stats_file = os.path.join(graphs_dir, "stats.txt")
    with open(stats_file, "w") as f:
        
        portfolio_returns = data['Portfolio_Return']
        total_return = portfolio_returns.sum()
        portfolio_std_dev = portfolio_returns.std()

        initial_portfolio_value = 0
        final_portfolio_value = 0

        with open(stats_file, "w") as f:
            f.write("===== Portfolio Statistics =====\n")

            for col in data.columns:
                if col not in ['Year', 'Month', 'Portfolio_Return']:  # Skip non-asset columns
                    asset_name = col
                    
                    shares = shares_data[shares_data['Asset'] == asset_name]['Shares'].values[0]
                    
                    start_price = asset_prices_data[
                        (asset_prices_data['Year'] == data['Year'].iloc[0]) & 
                        (asset_prices_data['Month'] == data['Month'].iloc[0])
                    ][asset_name].values[0]
                    
                    end_price = asset_prices_data[
                        (asset_prices_data['Year'] == data['Year'].iloc[-1]) & 
                        (asset_prices_data['Month'] == data['Month'].iloc[-1])
                    ][asset_name].values[0]
                    
                    capital_allocated = shares * (end_price - start_price)
                    percentage_return = ((end_price - start_price) / start_price) * 100
                    
                    initial_portfolio_value += shares * start_price
                    final_portfolio_value += shares * end_price

                    f.write(f"Asset: {asset_name}\n")
                    f.write(f"  Start Price: {start_price:.2f}\n")
                    f.write(f"  End Price: {end_price:.2f}\n")
                    f.write(f"  Shares Held: {shares}\n")
                    f.write(f"  Capital Allocated: {capital_allocated:,.2f}\n")
                    f.write(f"  Percentage Return: {percentage_return:.2f}%\n\n")
                
                total_return_final = ((final_portfolio_value - initial_portfolio_value) / (initial_portfolio_value+0.0000001)) * 100

            f.write("===== Final Portfolio Statistics =====\n")
            f.write(f"Initial Portfolio Value: {initial_portfolio_value:,.2f}\n")
            f.write(f"Final Portfolio Value: {final_portfolio_value:,.2f}\n")
            f.write(f"Total Portfolio Return: {total_return_final:.2f}%\n")
            f.write(f"Portfolio Standard Deviation: {portfolio_std_dev:.2f}\n")

        print(f"Statistics written to {stats_file}")

def plot_cumulative_returns():
    cumulative_returns = data['Portfolio_Return']

    avg_return = float(data['Portfolio_Return'].mean())
    std_dev = float(data['Portfolio_Return'].std())

    plus_line= avg_return + std_dev
    minus_line = avg_return - std_dev

    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_returns, label="Monthly Returns", linewidth=2, color='blue')
    plt.axhline(y=avg_return, color='green', linestyle="--", label="Average Return")
    plt.axhline(y=plus_line, color='red', linestyle=":", label="Mean + Std Dev")
    plt.axhline(y=minus_line, color='red', linestyle=":", label="Mean - Std Dev")
    plt.title("Overall Monthly Returns")
    plt.xlabel("Months")
    plt.ylabel("Monthly Returns")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(graphs_dir, "monthly_returns.png"))
    plt.close()
    print("Monthly returns graph saved.")

def plot_asset_returns():
    asset_returns = data[assets]
    asset_returns.plot(figsize=(12, 6))
    plt.title("Asset-wise Monthly Returns")
    plt.xlabel("Months")
    plt.ylabel("Monthly Return")
    plt.legend(asset_returns.columns)
    plt.grid()
    plt.savefig(os.path.join(graphs_dir, "asset_returns.png"))
    plt.close()
    print("Asset-wise returns graph saved.")

def compare_sectoral_performance():
    sector_data = pd.DataFrame({
        sector: data[asset] for asset, sector in zip(assets, sectors)
    })
    sector_means = sector_data.mean()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=sector_means.index, y=sector_means.values, palette="viridis")
    plt.title("Sectoral Monthly Returns Comparison")
    plt.ylabel("Average Monthly Return")
    plt.xlabel("Sectors")
    plt.xticks(rotation=45)
    plt.grid()
    plt.savefig(os.path.join(graphs_dir, "sectoral_returns.png"))
    plt.close()
    print("Sectoral returns graph saved.")

if __name__ == "__main__":
    analyze_results()
    plot_cumulative_returns()
    plot_asset_returns()
    compare_sectoral_performance()
