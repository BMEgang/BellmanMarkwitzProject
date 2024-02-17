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


def allocate_weights_best_nn(data, window_size):
    total_returns = data.pct_change().rolling(window=window_size).apply(lambda x: (1 + x).prod() - 1).iloc[-1]
    positive_returns = total_returns[total_returns > 0]

    weights = np.zeros(len(data.columns))

    if not positive_returns.empty:
        # 获取正收益股票在数据框列中的位置
        selected_indices = [list(data.columns).index(stock) for stock in positive_returns.index]

        # 在有正收益的股票上分配权重
        selected_weights = positive_returns / positive_returns.sum()
        weights[selected_indices] = selected_weights.values
    else:
        # 如果没有股票有正收益，则均匀分配权重
        weights = np.full(len(data.columns), 1.0 / len(data.columns))

    return weights


def simulate_investment_best_nn(data, initial_capital, window_size):
    capital = initial_capital
    capital_over_time = []
    current_weights = np.zeros(len(data.columns))

    for i in range(window_size, len(data)):
        if i % window_size == 0:
            current_weights = allocate_weights_best_nn(data.iloc[:i], window_size)

        daily_returns = data.iloc[i] / data.iloc[i - 1] - 1
        capital *= (1 + np.dot(daily_returns, current_weights))
        capital_over_time.append(capital)

    return np.array(capital_over_time)


# 设置参数
window_size = 60  # 历史窗口大小

# 在交易数据集上运行 Best-N-of-N 模型
portfolio_values_bnn = simulate_investment_best_nn(trade, initial_capital, window_size)

# 绘制投资组合价值变化
plt.figure(figsize=(12, 6))
plt.plot(trade.index[window_size:], portfolio_values_bnn)
plt.title("Best-N-of-N Model Portfolio Value Over Time")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.savefig(f"{image_folder}/Best_NN_Model.png")

# 保存投资组合价值到 CSV
portfolio_df_bnn = pd.DataFrame({
    'Date': trade.index[window_size:],
    'Portfolio_Value': portfolio_values_bnn
})
csv_save_path_bnn = os.path.join(data_folder, "Best_NN_Model.csv")
portfolio_df_bnn.to_csv(csv_save_path_bnn, index=False)
