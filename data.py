# Define assets and sectors
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
sectors = list(assets_sectors.values())

# Constraints
max_risk = 0.10  # Max portfolio risk (standard deviation)
max_weight = 0.25  # Max weight for any single asset

total_capital = 10000000  # 10,000,000 INR

