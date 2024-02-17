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


def find_similar_pattern(data, window_size):
    # 计算历史窗口内的日收益率
    historical_returns = data.pct_change().rolling(window=window_size).mean().dropna()

    # 检查历史收益率是否为空
    if historical_returns.empty:
        return np.zeros(data.shape[1])

    # 寻找与最近时间窗口最相似的历史模式
    recent_pattern = data.pct_change().iloc[-window_size:].mean()
    similarities = historical_returns.apply(
        lambda x: np.dot(x, recent_pattern) / (np.linalg.norm(x) * np.linalg.norm(recent_pattern)), axis=1)
    most_similar_date = similarities.idxmax()
    return data.loc[most_similar_date]

# def simulate_investment_corn(data, initial_capital, window_sizes):
#     capital = initial_capital
#     capital_over_time = []
#     current_weights = np.zeros(len(data.columns))
#
#     for i in range(max_window_size, len(data)):
#         # 确保有足够的历史数据来形成一个完整的窗口
#         similar_patterns = [find_similar_pattern(data.iloc[:i], window) for window in window_sizes]
#         average_pattern = np.mean(similar_patterns, axis=0)
#
#         # 将平均模式转换为权重
#         current_weights = average_pattern / average_pattern.sum()
#
#         daily_returns = data.iloc[i] / data.iloc[i - 1] - 1
#         capital *= (1 + np.dot(daily_returns, current_weights))
#         capital_over_time.append(capital)
#
#     return np.array(capital_over_time)
def simulate_investment_corn(data, initial_capital, window_sizes):
    capital = initial_capital
    capital_over_time = []
    weights_over_time = []  # 存储每日权重
    max_window_size = max(window_sizes)

    for i in range(max_window_size, len(data)):
        similar_patterns = [find_similar_pattern(data.iloc[:i], window) for window in window_sizes]
        average_pattern = np.mean(similar_patterns, axis=0)
        current_weights = average_pattern / average_pattern.sum()
        weights_over_time.append(current_weights)  # 存储每日权重

        daily_returns = data.iloc[i] / data.iloc[i - 1] - 1
        capital *= (1 + np.dot(daily_returns, current_weights))
        capital_over_time.append(capital)

    return np.array(capital_over_time), np.array(weights_over_time)



# 设置参数
window_sizes = [10, 20, 30]  # 不同的历史窗口大小
max_window_size = max(window_sizes)

# 在交易数据集上运行 CORN 模型
# portfolio_values_corn = simulate_investment_corn(trade, initial_capital, window_sizes)
portfolio_values_corn, daily_weights_corn = simulate_investment_corn(trade, initial_capital, window_sizes)

# 绘制投资组合价值变化
plt.figure(figsize=(12, 6))
plt.plot(trade.index[max(window_sizes):], portfolio_values_corn)
plt.title("CORN Model Portfolio Value Over Time")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.savefig(f"{image_folder}/CORN_Model.png")

# 保存投资组合价值到 CSV
portfolio_df_corn = pd.DataFrame({
    'Date': trade.index[max(window_sizes):],
    'Portfolio_Value': portfolio_values_corn
})
csv_save_path_corn = os.path.join(data_folder, "CORN_Model.csv")
portfolio_df_corn.to_csv(csv_save_path_corn, index=False)

# 筛选2017年11月至2017年12月的数据
start_date = "2017-11-01"
end_date = "2017-12-01"

# 获取特定时间段的权重和资本
selected_dates = (trade.index[max_window_size:] >= start_date) & (trade.index[max_window_size:] <= end_date)
selected_weights = daily_weights_corn[selected_dates]
average_weights = np.mean(selected_weights, axis=0)

# 获取特定时间段的初始和最终资本
initial_capital_period = portfolio_values_corn[trade.index[max_window_size:].searchsorted(start_date)]
final_capital_period = portfolio_values_corn[trade.index[max_window_size:].searchsorted(end_date) - 1]

# 计算收益
absolute_return = final_capital_period - initial_capital_period
relative_return = absolute_return / initial_capital_period

# 打印结果
print("Average Weights from 2017-11 to 2017-12:")
print(average_weights)
print("\nAbsolute Return from 2017-11 to 2017-12: ${:.2f}".format(absolute_return))
print("Relative Return from 2017-11 to 2017-12: {:.2%}".format(relative_return))

# 计算特定时间段内的日收益率
selected_returns = portfolio_values_corn[1:] / portfolio_values_corn[:-1] - 1
selected_returns_period = selected_returns[(trade.index[max_window_size:-1] >= start_date) & (trade.index[max_window_size:-1] <= end_date)]

# 计算波动率（标准差）
volatility = np.std(selected_returns_period)

# 打印风险（波动率）
print("Volatility from 2017-11 to 2017-12: {:.2%}".format(volatility))