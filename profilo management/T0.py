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

def simulate_investment_t0(data, initial_capital):
    capital = initial_capital
    capital_over_time = []
    selected_stocks = []  # 用于存储每天选择的股票
    num_stocks = len(data.columns)

    for i in range(1, len(data)):
        cumulative_returns = data.iloc[:i].pct_change().sum()
        best_stock = cumulative_returns.idxmax()
        selected_stocks.append(best_stock)  # 记录选择的股票

        today_return = data.iloc[i][best_stock] / data.iloc[i - 1][best_stock] - 1
        capital *= (1 + today_return)
        capital_over_time.append(capital)

    return np.array(capital_over_time), selected_stocks

# 运行T0模型
# portfolio_values_t0 = simulate_investment_t0(trade, initial_capital)
portfolio_values_t0, selected_stocks_t0 = simulate_investment_t0(trade, initial_capital)

# 选择特定时间段
start_date = "2017-11-01"
end_date = "2017-12-01"
selected_period = (trade.index[1:] >= start_date) & (trade.index[1:] <= end_date)
selected_stocks_period = [stock for i, stock in enumerate(selected_stocks_t0) if selected_period[i]]

# 计算配置比例
stock_allocation = pd.Series(selected_stocks_period).value_counts(normalize=True)

# 输出配置比例
print("Stock Allocation from 2017-11 to 2017-12:")
print(stock_allocation)

# 绘图
plt.figure(figsize=(12, 6))
plt.plot(trade.index[1:], portfolio_values_t0)
plt.title("T0 Model Portfolio Value Over Time")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.savefig(os.path.join(image_folder, "T0_Model.png"))

# 保存结果
portfolio_df_t0 = pd.DataFrame({'Date': trade.index[1:], 'Portfolio_Value': portfolio_values_t0})
csv_save_path_t0 = os.path.join(data_folder, "T0_Model.csv")
portfolio_df_t0.to_csv(csv_save_path_t0, index=False)

# 计算特定时间段内的初始和最终资本
initial_capital_period = portfolio_values_t0[trade.index[1:].searchsorted(start_date)]
final_capital_period = portfolio_values_t0[trade.index[1:].searchsorted(end_date)]

# 计算收益
absolute_return = final_capital_period - initial_capital_period
relative_return = absolute_return / initial_capital_period

# 输出收益
print("\nAbsolute Return from 2017-11 to 2017-12: ${:.2f}".format(absolute_return))
print("Relative Return from 2017-11 to 2017-12: {:.2%}".format(relative_return))


# 计算特定时间段内的日收益率
selected_returns = portfolio_values_t0[1:] / portfolio_values_t0[:-1] - 1
selected_returns_period = selected_returns[(trade.index[1:-1] >= start_date) & (trade.index[1:-1] <= end_date)]

# 计算波动率（标准差）
volatility = np.std(selected_returns_period)

# 打印风险（波动率）
print("Volatility from 2017-11 to 2017-12: {:.2%}".format(volatility))