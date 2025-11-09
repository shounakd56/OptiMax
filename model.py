import pandas as pd
import numpy as np
from helper import (
    optimize_slsqp,
    optimize_qp,
    optimize_de,
    calculate_cumulative_returns
)
from data import max_risk, max_weight, assets, total_capital
import os

file_path = "asset_prices.csv"
data = pd.read_csv(file_path)

returns_columns = [col for col in data.columns if "_Return" in col]
price_columns = [col for col in data.columns if col not in ['Year', 'Month'] + returns_columns]

monthly_returns = data[returns_columns]
monthly_prices = data[price_columns]

optimization_returns = monthly_returns.loc[data['Year'] < 2022]
backtesting_prices = monthly_prices.loc[data['Year'] >= 2022]
backtesting_returns = monthly_returns.loc[data['Year'] >= 2022]

mean_returns = optimization_returns.mean()
cov_matrix = optimization_returns.cov()

def run_optimization(method="SLSQP"):
    if method == "SLSQP":
        return optimize_slsqp(mean_returns, cov_matrix, max_risk, max_weight)
    elif method == "QP":
        return optimize_qp(mean_returns, cov_matrix,max_weight)
    elif method == "DE":
        return optimize_de(mean_returns, cov_matrix, max_risk, max_weight)
    else:
        raise ValueError("Invalid optimization method.")

def calculate_shares(weights, prices):
    shares = {}
    for asset, weight in zip(assets, weights):
        capital_allocated = total_capital * weight
        price = prices[asset].iloc[0]  # First month price
        shares[asset] = capital_allocated // price
    return shares

def backtest_portfolio(weights):
    portfolio_returns = backtesting_returns @ weights
    cumulative_returns = calculate_cumulative_returns(portfolio_returns)

    shares = calculate_shares(weights, backtesting_prices)

    monthly_data = pd.DataFrame({
        "Year": data.loc[data['Year'] >= 2022, 'Year'],
        "Month": data.loc[data['Year'] >= 2022, 'Month'],
        "Portfolio_Return": portfolio_returns,
    })
    for asset in assets:
        asset_ret_column = asset + "_Return"
        monthly_data[asset] = backtesting_returns[asset_ret_column]

    model_results_file = "model_results.csv"
    if not os.path.exists(model_results_file):
        monthly_data.to_csv(model_results_file, index=False)
    else:
        monthly_data.to_csv(model_results_file, mode='a', header=False, index=False)

    share_data = pd.DataFrame({
        "Asset": assets,
        "Shares": [shares[asset] for asset in assets]
    })
    share_data.to_csv("shares_allocation.csv", index=False)
    cumulative_returns.plot(title="Cumulative Returns (Backtesting Period)", figsize=(10, 6))

    # start_prices = backtesting_prices.iloc[0].values
    # end_prices = backtesting_prices.iloc[-1].values
    portfolio_value = sum([shares[asset] * data[
                        (data['Year'] == data['Year'].iloc[-1]) & 
                        (data['Month'] == data['Month'].iloc[-1])
                    ][asset].values[0] for i, asset in enumerate(assets)])
    print(f"Total Portfolio Value at End: ₹{portfolio_value:,.2f}")

if __name__ == "__main__":
    method = "QP"  # Change to "SLSQP" or "QP" or "DE" for other optimizations
    weights, optimized_return, optimized_risk = run_optimization(method=method)

    print("Optimal Weights:", weights)
    print("Optimized Return:", optimized_return)
    print("Optimized Risk:", optimized_risk)

    backtest_portfolio(weights)
