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

print(trade.tail())

# ONS模型参数初始化
initial_capital = 10000
# weights = np.full(len(successful_tickers), 1.0 / len(successful_tickers))  # 初始等权重
# learning_rate = 0.01  # 学习率

def anticor_update(data, window_short, window_long):
    # 计算两个时间窗口的收益率
    returns_short = data[-window_short:].pct_change().dropna()
    returns_long = data[-window_long:].pct_change().dropna()

    # 计算两个窗口的相关性
    correlation = returns_short.corrwith(returns_long, axis=0)

    # 生成权重：对负相关的资产加权，对正相关的资产减权
    weights = 1 - correlation
    weights[weights < 0] = 0  # 确保权重非负
    return weights / weights.sum()  # 归一化权重

def simulate_investment_anticor(data, initial_capital, window_short, window_long):
    capital = initial_capital
    capital_over_time = []
    weights = np.full(len(data.columns), 1.0 / len(data.columns))  # 初始等权重

    for i in range(window_long, len(data)):
        current_data = data.iloc[:i]
        weights = anticor_update(current_data, window_short, window_long)
        daily_returns = data.iloc[i] / data.iloc[i - 1] - 1
        capital *= (1 + np.dot(daily_returns, weights))
        capital_over_time.append(capital)

    return np.array(capital_over_time)

# 选择时间窗口
window_short = 30  # 短期窗口，例如30天
window_long = 90  # 长期窗口，例如90天

# 在交易数据集上运行 Anticor 模型
portfolio_values_anticor = simulate_investment_anticor(trade, initial_capital, window_short, window_long)

# 绘制投资组合价值变化
plt.figure(figsize=(12, 6))
plt.plot(trade.index[window_long:], portfolio_values_anticor)
plt.title("Anticor Model Portfolio Value Over Time")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.savefig(f"{image_folder}/Anticor_Model.png")

# 保存投资组合价值到 CSV
portfolio_df_anticor = pd.DataFrame({
    'Date': trade.index[window_long:],
    'Portfolio_Value': portfolio_values_anticor
})
csv_save_path_anticor = os.path.join(data_folder, "Anticor_Model.csv")
portfolio_df_anticor.to_csv(csv_save_path_anticor, index=False)
