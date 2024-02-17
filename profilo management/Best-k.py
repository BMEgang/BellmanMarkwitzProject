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

def select_best_k_stocks(data, window_size, k):
    # 计算每只股票在指定窗口期内的总收益率
    total_returns = data.pct_change().rolling(window=window_size).apply(lambda x: (1 + x).prod() - 1).iloc[-1]

    # 选择表现最佳的 k 只股票
    best_k_stocks = total_returns.nlargest(k).index
    return best_k_stocks

# def simulate_investment_best_k(data, initial_capital, window_size, k):
#     capital = initial_capital
#     capital_over_time = []
#     current_weights = np.zeros(len(data.columns))
#
#     for i in range(window_size, len(data)):
#         if i % window_size == 0:  # 每隔 window_size 天，重新选择股票和分配权重
#             selected_stocks = select_best_k_stocks(data.iloc[:i], window_size, k)
#             current_weights = np.where(data.columns.isin(selected_stocks), 1.0 / k, 0)
#
#         daily_returns = data.iloc[i] / data.iloc[i - 1] - 1
#         capital *= (1 + np.dot(daily_returns, current_weights))
#         capital_over_time.append(capital)
#
#     return np.array(capital_over_time)
def simulate_investment_best_k(data, initial_capital, window_size, k):
    capital = initial_capital
    capital_over_time = []
    current_weights = np.zeros(len(data.columns))
    stock_selection_over_time = []  # 存储每个窗口期的股票选择

    for i in range(window_size, len(data)):
        if i % window_size == 0:  # 每隔 window_size 天，重新选择股票和分配权重
            selected_stocks = select_best_k_stocks(data.iloc[:i], window_size, k)
            current_weights = np.where(data.columns.isin(selected_stocks), 1.0 / k, 0)
            stock_selection_over_time.append(selected_stocks)  # 存储股票选择

        daily_returns = data.iloc[i] / data.iloc[i - 1] - 1
        capital *= (1 + np.dot(daily_returns, current_weights))
        capital_over_time.append(capital)

    return np.array(capital_over_time), stock_selection_over_time

# 设置参数
window_size = 60  # 历史窗口大小
k = 5  # 选择的股票数量

# 在交易数据集上运行 Best-k 模型
# portfolio_values_bk = simulate_investment_best_k(trade, initial_capital, window_size, k)
portfolio_values_bk, stock_selection_bk = simulate_investment_best_k(trade, initial_capital, window_size, k)

# 绘制投资组合价值变化
plt.figure(figsize=(12, 6))
plt.plot(trade.index[window_size:], portfolio_values_bk)
plt.title("Best-k Model Portfolio Value Over Time")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.savefig(f"{image_folder}/Best_k_Model.png")

# 保存投资组合价值到 CSV
portfolio_df_bk = pd.DataFrame({
    'Date': trade.index[window_size:],
    'Portfolio_Value': portfolio_values_bk
})
csv_save_path_bk = os.path.join(data_folder, "Best_k_Model.csv")
portfolio_df_bk.to_csv(csv_save_path_bk, index=False)

# 计算选中股票的频率
stock_frequency = pd.Series([stock for selection in stock_selection_bk for stock in selection]).value_counts()

# 打印股票频率
print("Stock Selection Frequency:")
print(stock_frequency)

# 筛选2017年11月至2017年12月的数据
start_date = "2017-11-01"
end_date = "2017-12-01"

# 获取特定时间段的初始和最终资本
initial_capital_period = portfolio_values_bk[trade.index[window_size:].searchsorted(start_date)]
final_capital_period = portfolio_values_bk[trade.index[window_size:].searchsorted(end_date) - 1]

# 计算收益
absolute_return = final_capital_period - initial_capital_period
relative_return = absolute_return / initial_capital_period

# 打印结果
print("\nAbsolute Return from 2017-11 to 2017-12: ${:.2f}".format(absolute_return))
print("Relative Return from 2017-11 to 2017-12: {:.2%}".format(relative_return))
