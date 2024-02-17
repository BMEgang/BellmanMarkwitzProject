import yfinance as yf
import numpy as np
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
from finrl import config_tickers

# 获取当前时间并格式化为字符串
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# 基础路径和文件夹设置
base_folder = "/Users/ganghu/Desktop/pythonProject1/final"
image_folder = os.path.join(base_folder, f"{current_time}/image")
model_folder = os.path.join(base_folder, f"{current_time}/model")
data_folder = os.path.join(base_folder, f"{current_time}/data")
result_folder = os.path.join(base_folder, f"{current_time}/result")

# 创建必要的文件夹
for folder in [image_folder, model_folder, data_folder, result_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# 获取股票数据
tickers = config_tickers.DOW_30_TICKER
successful_tickers = []
for ticker in tickers:
    try:
        data = yf.download(ticker, start="2008-12-20", end="2018-08-30")
        if not data.empty:
            successful_tickers.append(ticker)
    except Exception as e:
        print(f"Failed to download {ticker}: {e}")

data_combined = yf.download(successful_tickers, start="2008-12-20", end="2018-09-30")["Close"]

# 数据集划分
train = data_combined["2009-01-01":"2015-01-01"]
test = data_combined["2015-01-01":"2016-01-01"]
trade = data_combined["2016-01-01":]

# M0模型参数
initial_capital = 10000

def simulate_investment_m0(data, initial_capital):
    capital = initial_capital
    capital_over_time = []
    num_stocks = len(data.columns)
    weights = np.full(num_stocks, 1.0 / num_stocks)

    for i in range(1, len(data)):
        returns = data.iloc[i] / data.iloc[i - 1] - 1
        capital *= (1 + np.dot(returns, weights))
        capital_over_time.append(capital)

    return np.array(capital_over_time)

# 运行M0模型
portfolio_values_m0 = simulate_investment_m0(trade, initial_capital)

# 绘图
plt.figure(figsize=(12, 6))
plt.plot(trade.index[1:], portfolio_values_m0)
plt.title("M0 Model Portfolio Value Over Time")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.savefig(os.path.join(image_folder, "M0_Model.png"))

# 保存结果
portfolio_df_m0 = pd.DataFrame({'Date': trade.index[1:], 'Portfolio_Value': portfolio_values_m0})
csv_save_path_m0 = os.path.join(data_folder, "M0_Model.csv")
portfolio_df_m0.to_csv(csv_save_path_m0, index=False)
