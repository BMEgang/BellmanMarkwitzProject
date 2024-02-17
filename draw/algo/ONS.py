import yfinance as yf
import numpy as np
import pandas as pd
from finrl import config_tickers
import os
from datetime import datetime
import matplotlib.pyplot as plt

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
        data = yf.download(ticker, start="2008-12-20", end="2018-08-30")
        if not data.empty:
            successful_tickers.append(ticker)
    except Exception as e:
        print(f"Failed to download {ticker}: {e}")

# 下载数据
data_combined = yf.download(successful_tickers, start="2008-12-20", end="2018-09-30")["Close"]

# 数据集划分
train = data_combined["2009-01-01":"2015-01-01"]
test = data_combined["2015-01-01":"2016-01-01"]
trade = data_combined["2016-01-01":]

# ONS模型参数初始化
initial_capital = 10000
weights = np.full(len(successful_tickers), 1.0 / len(successful_tickers))  # 初始等权重
learning_rate = 0.01  # 学习率


def ons_update(weights, returns, learning_rate):
    gradient = returns / (np.dot(returns, weights) + 1e-8)  # 防止除以0
    clipped_gradient = np.clip(gradient, a_min=-0.1, a_max=0.1)  # 裁剪梯度以防溢出
    weights *= np.exp(learning_rate * clipped_gradient)
    return weights / np.sum(weights)

# 模拟投资
def simulate_investment(data, initial_capital, weights, learning_rate):
    capital = initial_capital
    capital_over_time = []

    for i in range(1, len(data)):
        returns = data.iloc[i] / data.iloc[i - 1] - 1
        weights = ons_update(weights, returns, learning_rate)
        capital *= (1 + np.sum(returns * weights))
        capital_over_time.append(capital)

    return np.array(capital_over_time)

# 在交易数据集上运行模型
portfolio_values = simulate_investment(trade, initial_capital, weights, learning_rate)

# 绘制投资组合价值变化
plt.figure(figsize=(12, 6))
plt.plot(trade.index[1:], portfolio_values)
plt.title("Portfolio Value Over Time")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.savefig(f"{image_folder}/ONS.png")

# 创建一个 DataFrame 来存储日期和投资组合价值
portfolio_df = pd.DataFrame({
    'Date': trade.index[1:],
    'Portfolio_Value': portfolio_values
})

# 指定 CSV 文件的保存路径
csv_save_path = os.path.join(data_folder, "ONS.csv")

# 保存到 CSV
portfolio_df.to_csv(csv_save_path, index=False)