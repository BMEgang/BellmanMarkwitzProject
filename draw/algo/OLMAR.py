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

def olmar_update(weights, prices, moving_average, epsilon=10):
    """
    OLMAR 更新规则
    :param weights: 当前的投资组合权重
    :param prices: 当天的资产价格
    :param moving_average: 资产的移动平均价格
    :param epsilon: 阈值，决定权重调整的幅度
    :return: 更新后的权重
    """
    relative_prices = moving_average / prices
    x_bar = np.dot(weights, relative_prices)
    lam = max(0, (epsilon - x_bar) / np.sum((relative_prices - x_bar)**2))
    weights = weights * lam * (relative_prices - x_bar) + weights
    weights = np.maximum(weights, 0)  # 确保权重非负
    weights /= np.sum(weights)  # 归一化权重
    return weights

def simulate_investment_olmar(data, initial_capital, initial_weights, window_size, epsilon):
    capital = initial_capital
    capital_over_time = []
    weights = np.copy(initial_weights)

    for i in range(window_size, len(data)):
        current_prices = data.iloc[i]
        moving_average = data.iloc[i-window_size:i].mean()
        weights = olmar_update(weights, current_prices, moving_average, epsilon)
        daily_returns = data.iloc[i] / data.iloc[i - 1] - 1
        capital *= (1 + np.dot(daily_returns, weights))
        capital_over_time.append(capital)

    return np.array(capital_over_time)

# 初始化参数
window_size = 5  # 移动平均线窗口大小
epsilon = 10  # OLMAR 阈值

# 在交易数据集上运行 OLMAR 模型
portfolio_values_olmar = simulate_investment_olmar(trade, initial_capital, weights, window_size, epsilon)

# 绘制投资组合价值变化
plt.figure(figsize=(12, 6))
plt.plot(trade.index[window_size:], portfolio_values_olmar)
plt.title("OLMAR Model Portfolio Value Over Time")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.savefig(f"{image_folder}/OLMAR_Model.png")

# 保存投资组合价值到 CSV
portfolio_df_olmar = pd.DataFrame({
    'Date': trade.index[window_size:],
    'Portfolio_Value': portfolio_values_olmar
})
csv_save_path_olmar = os.path.join(data_folder, "OLMAR_Model.csv")
portfolio_df_olmar.to_csv(csv_save_path_olmar, index=False)
