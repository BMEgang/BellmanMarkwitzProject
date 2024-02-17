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

def simulate_investment(data, initial_capital, weights, learning_rate):
    capital = initial_capital
    capital_over_time = []

    for i in range(1, len(data)):
        # 计算每个交易日的资产收益率
        returns = data.iloc[i] / data.iloc[i - 1] - 1
        # 计算投资组合的整体收益率
        portfolio_return = np.dot(returns, weights)
        # 更新资本
        capital *= (1 + portfolio_return)
        # 记录资本
        capital_over_time.append(capital)

    return np.array(capital_over_time)


def select_stock_pool(data, num_stocks=10):
    # 示例：选择历史波动性最低的股票
    volatilities = data.pct_change().std()
    selected_stocks = volatilities.nsmallest(num_stocks).index
    return selected_stocks

def allocate_weights_equal(stock_pool):
    num_stocks = len(stock_pool)
    weights = np.full(num_stocks, 1.0 / num_stocks)
    return weights

# 选择股票池
stock_pool = select_stock_pool(data_combined)

# 为股票池分配权重
weights_sp = allocate_weights_equal(stock_pool)

# 从原始数据中提取股票池数据，并与交易数据对齐
data_pool = data_combined[stock_pool].loc[trade.index]


# 模拟投资
portfolio_values_sp = simulate_investment(data_pool, initial_capital, weights_sp, learning_rate=0)  # 无学习率，因为权重固定

# 绘制投资组合价值变化
plt.figure(figsize=(12, 6))
plt.plot(data_pool.index[1:], portfolio_values_sp)
plt.title("SP Model Portfolio Value Over Time")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.savefig(f"{image_folder}/SP_Model.png")

# 保存投资组合价值到 CSV
portfolio_df_sp = pd.DataFrame({
    'Date': data_pool.index[1:],
    'Portfolio_Value': portfolio_values_sp
})
csv_save_path_sp = os.path.join(data_folder, "SP_Model.csv")
portfolio_df_sp.to_csv(csv_save_path_sp, index=False)

# 筛选2017年11月至2017年12月的数据
start_date = "2017-11-01"
end_date = "2017-12-01"

# 获取特定时间段的初始和最终资本
initial_capital_period = portfolio_values_sp[data_pool.index.searchsorted(start_date)]
final_capital_period = portfolio_values_sp[data_pool.index.searchsorted(end_date) - 1]  # 减去1因为索引是从0开始的

# 计算收益
absolute_return = final_capital_period - initial_capital_period
relative_return = absolute_return / initial_capital_period


# 打印股票池中的股票代码
print("Selected stocks for the portfolio:")
print(stock_pool)
# 打印结果
print("Average Weights from 2017-11 to 2017-12:")
print(weights_sp)  # 平均比例已由 weights_sp 确定
print("\nAbsolute Return from 2017-11 to 2017-12: ${:.2f}".format(absolute_return))
print("Relative Return from 2017-11 to 2017-12: {:.2%}".format(relative_return))

# 计算投资组合的每日收益率
portfolio_returns_sp = np.diff(portfolio_values_sp) / portfolio_values_sp[:-1]

# 选取特定时间段内的每日收益率
selected_returns_sp = portfolio_returns_sp[(data_pool.index[1:-1] >= start_date) & (data_pool.index[1:-1] <= end_date)]

# 计算波动率（标准差）
volatility_sp = np.std(selected_returns_sp)

# 输出风险（波动率）
print("Volatility from 2017-11 to 2017-12: {:.2%}".format(volatility_sp))
