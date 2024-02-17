import os
import pandas as pd
import yfinance as yf
from finrl import config_tickers
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Basic settings
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
base_folder = "/Users/ganghu/Desktop/pythonProject1/final"
image_folder = os.path.join(base_folder, f"{current_time}/image")
data_folder = os.path.join(base_folder, f"{current_time}/data")

# Create directories
for folder in [image_folder, data_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"make {folder} success")

# Download stock data
tickers = config_tickers.DOW_30_TICKER
# 尝试下载数据，如果失败则从列表中移除对应的股票代码
successful_tickers = []
for ticker in tickers:
    try:
        data = yf.download(ticker, start="2008-12-20", end="2018-08-30")
        if not data.empty:
            successful_tickers.append(ticker)
    except Exception as e:
        print(f"Failed to download {ticker}: {e}")

data = yf.download(successful_tickers, start="2016-01-01", end="2018-09-30")['Close']

# Initial capital
initial_capital = 10000

# Buy-and-Hold: Calculate initial quantities
initial_prices = data.iloc[0]
initial_quantities = (initial_capital/len(successful_tickers)) / initial_prices

# Portfolio value over time
portfolio_values = (data * initial_quantities).sum(axis=1)

# Create and save portfolio value DataFrame
capital_curve_df = pd.DataFrame({'Date': data.index, 'Capital': portfolio_values})
capital_curve_df.set_index('Date', inplace=True)

# Plotting the Capital Curve
plt.figure(figsize=(12, 6))
plt.plot(capital_curve_df.index, capital_curve_df['Capital'], label='Capital Over Time')
plt.title('BAH Model Capital Curve')
plt.xlabel('Date')
plt.ylabel('Capital in USD')
plt.legend()
plt.savefig(f"{image_folder}/BAH_Model_Capital_Curve.png")

# Save Capital Curve Data
capital_curve_df.to_csv(f"{data_folder}/BAH_capital_curve.csv")
