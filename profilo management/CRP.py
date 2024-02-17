import os
import pandas as pd
import yfinance as yf
from finrl import config_tickers
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime

# 获取当前时间并格式化为字符串（例如 '2023-11-04_12-30-00'）
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# ***************************************** knowledge distillation **************
# 基础路径
base_folder = "/Users/ganghu/Desktop/pythonProject1/final"
# 创建带有当前时间的文件夹路径
image_folder = os.path.join(base_folder, f"{current_time}/image")
model_folder = os.path.join(base_folder, f"{current_time}/model")
data_folder = os.path.join(base_folder, f"{current_time}/data")
result_folder = os.path.join(base_folder, f"{current_time}/result")

# 检查并创建图像文件夹
if not os.path.exists(image_folder):
    print(f"make {image_folder} success")
    os.makedirs(image_folder)

# 检查并创建模型文件夹
if not os.path.exists(model_folder):
    print(f"make {model_folder} success")
    os.makedirs(model_folder)

# 检查并创建数据文件夹
if not os.path.exists(data_folder):
    print(f"make {data_folder} success")
    os.makedirs(data_folder)

if not os.path.exists(result_folder):
    print(f"make {result_folder} success")
    os.makedirs(result_folder)

# 股票代码
tickers = config_tickers.DOW_30_TICKER
# 尝试下载数据，如果失败则从列表中移除对应的股票代码
successful_tickers = []
for ticker in tickers:
    try:
        data = yf.download(ticker, start="2016-01-01", end="2018-08-30")
        if not data.empty:
            successful_tickers.append(ticker)
    except Exception as e:
        print(f"Failed to download {ticker}: {e}")

# 数据收集
data = yf.download(successful_tickers, start="2016-01-01", end="2018-09-30")['Close']

dates_array = data.index.to_numpy()
price = data.to_numpy()

# 假设每个股票的目标权重都是相等的
# Number of successful tickers
num_tickers = len(successful_tickers)

# Calculate target weight
target_weight_value = 1.0 / num_tickers

# Create a NumPy array filled with the target weight for each ticker
target_weight = np.full(num_tickers, target_weight_value)

quantity = np.zeros_like(price)

# 初始资金
initial_investment = 10000

for i in range(price.shape[0]):
    if i == 0:
        quantity[i] = initial_investment * target_weight / price[i]
    else:
        portfolio_value = (quantity[i-1] * price[i]).sum()
        quantity[i] = portfolio_value * target_weight / price[i]

# Final assembly
# Correcting the columns MultiIndex to match the number of tickers
columns = pd.MultiIndex.from_product([["price", "quantity"], successful_tickers])

# Constructing the DataFrame
df = pd.DataFrame(np.hstack([price, quantity]), columns=columns)

# Calculate portfolio value
df["portfolio_value"] = (df["price"] * df["quantity"]).sum(axis=1)
print(df)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(dates_array, df["portfolio_value"], label='Portfolio Value Over Time')
plt.title('Portfolio Value Over Time')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gcf().autofmt_xdate()  # Auto-format the x-axis labels for readability
plt.legend()
plt.savefig(f"{image_folder}/Portfolio_Value.png")

# Creating a new DataFrame for saving
portfolio_data_to_save = pd.DataFrame({
    'Date': dates_array,
    'Portfolio_Value': df["portfolio_value"]
})

portfolio_data_to_save.to_csv(f"{data_folder}/CRP_portfolio_values.csv", index=False)