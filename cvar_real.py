import numpy as np
import pandas as pd
import cvxpy as cp
total_capital = 10000000
# Load or simulate historical returns
# np.random.seed(42)
# num_assets = 5
# num_samples = 1000
# returns = np.random.randn(num_samples, num_assets) * 0.01  # Simulated daily returns
# print(returns)

#Use historical data

file_path = "asset_prices.csv"
data = pd.read_csv(file_path)

assets_sectors = {
    'BAJFINANCE.NS': 'Financials',
    'HDFCBANK.NS': 'Financials',
    'HINDUNILVR.NS': 'Consumer Goods',
    'ICICIBANK.NS': 'Financials',
    'INFY.NS': 'IT',
    'ITC.NS': 'Consumer Goods',
    'RELIANCE.NS': 'Energy',
    'SBIN.NS': 'Financials',
    'TATAMOTORS.NS': 'Automobile',
    'TCS.NS': 'IT'
}
assets = list(assets_sectors.keys())
def calculate_shares(weights, prices):
    shares = {}
    for asset, weight in zip(assets, weights):
        capital_allocated = total_capital * weight
        price = prices[asset].iloc[0]  # First month price
        shares[asset] = capital_allocated // price
    return shares

assets = list(assets_sectors.keys())
sectors = list(assets_sectors.values())

num_assets = len(assets)
# print(num_assets)

returns_columns = [col for col in data.columns if "_Return" in col]
price_columns = [col for col in data.columns if col not in ['Year', 'Month'] + returns_columns]

monthly_returns = data[returns_columns]
monthly_prices = data[price_columns]

optimization_returns = monthly_returns.loc[data['Year'] < 2022]
optimization_returns = optimization_returns.dropna()
num_samples = optimization_returns.shape[0]
backtesting_prices = monthly_prices.loc[data['Year'] >= 2022]
backtesting_returns = monthly_returns.loc[data['Year'] >= 2022]
# print(optimization_returns)
mean_returns = optimization_returns.mean()
cov_matrix = optimization_returns.cov()
# Parameters
alpha = 0.95  # Confidence level for CVaR
investment_budget = 1  # Total investment

# Variables
w = cp.Variable(num_assets)  # Portfolio weights
z = cp.Variable(num_samples)  # Auxiliary variables for CVaR
VaR = cp.Variable()  # Value at Risk (VaR)

# print(num_assets)
# print(num_samples)
# print(optimization_returns.shape)

# print(w.shape)
# print(type(w))
optimization_returns = optimization_returns.to_numpy()
print(optimization_returns)
# Objective: Minimize CVaR
objective = cp.Minimize(VaR + (1 / (1 - alpha)) * cp.sum(z) / num_samples)

# Constraints
constraints = [
    cp.sum(w) == investment_budget,  # Full investment
    w >= 0,  # No short selling
    z >= 0,  # Non-negativity for auxiliary variables
    z >= -optimization_returns @ w - VaR,  # CVaR condition
]

objective_values = []  # Objective values during optimization
portfolio_weights = []  # Portfolio weights during optimization


# def capture_metrics():
#     """Helper function to capture metrics during optimization"""
#     objective_values.append(problem.value)  # Capture current objective value
#     portfolio_weights.append(w.value)  # Capture current portfolio weights


# Problem definition
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.ECOS, verbose=True)
shares=calculate_shares(w.value,backtesting_prices)
print(shares)
shares=assets = list(shares.values())
# import matplotlib.pyplot as plt
# plt.plot(objective_values)
# plt.xlabel("Iteration")
# plt.ylabel("Objective value")
# plt.title("Optimization progress")
# plt.show()

# print(np.array(backtesting_prices.iloc[-1]))
cov_back_mat = backtesting_returns.cov()
Final_portfolio_value = np.array(shares) @ (np.array(backtesting_prices.iloc[-1]))
return_port = (Final_portfolio_value-total_capital)/total_capital*100
print(return_port)
print(Final_portfolio_value)
print(w.value @ cov_back_mat @ w.value)

# Results
if problem.status == cp.OPTIMAL:
    print("Optimal portfolio weights:", w.value)
    print("Value at Risk (VaR):", VaR.value)
    print("Conditional Value at Risk (CVaR):", VaR.value + (1 / (1 - alpha)) * np.mean(z.value))
else:
    print("Problem could not be solved. Status:", problem.status)