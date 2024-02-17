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


def cwmr_update(weights, returns, epsilon=0.5, aggressiveness=0.1, confidence=0.5):
    """
    CWMR 更新规则
    :param weights: 当前的投资组合权重
    :param returns: 当天的资产收益率
    :param epsilon: 阈值，用于判断是否需要调整权重
    :param aggressiveness: 攻击性，决定权重调整的幅度
    :param confidence: 置信度，反映预测的不确定性
    :return: 更新后的权重
    """
    portfolio_return = np.dot(weights, returns)
    loss = max(0, epsilon - portfolio_return)
    norm_squared = np.sum(returns ** 2)

    if norm_squared > 0:
        tau = loss / norm_squared
        step = min(aggressiveness * confidence, tau)
        weights += step * returns
        weights = np.maximum(weights, 0)  # 确保权重非负
        weights /= np.sum(weights)  # 归一化权重
    return weights

def simulate_investment_cwmr(data, initial_capital, initial_weights, epsilon, aggressiveness, confidence):
    capital = initial_capital
    capital_over_time = []
    weights = np.copy(initial_weights)

    for i in range(1, len(data)):
        returns = data.iloc[i] / data.iloc[i - 1] - 1
        weights = cwmr_update(weights, returns, epsilon, aggressiveness, confidence)
        capital *= (1 + np.dot(returns, weights))
        capital_over_time.append(capital)

    return np.array(capital_over_time)

# 初始化参数
epsilon = 0.5  # CWMR 阈值
aggressiveness = 0.1  # CWMR 攻击性
confidence = 0.5  # CWMR 置信度

# 在交易数据集上运行 CWMR 模型
portfolio_values_cwmr = simulate_investment_cwmr(trade, initial_capital, weights, epsilon, aggressiveness, confidence)

# 绘制投资组合价值变化
plt.figure(figsize=(12, 6))
plt.plot(trade.index[1:], portfolio_values_cwmr)
plt.title("CWMR Model Portfolio Value Over Time")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.savefig(f"{image_folder}/CWMR_Model.png")

# 保存投资组合价值到 CSV
portfolio_df_cwmr = pd.DataFrame({
    'Date': trade.index[1:],
    'Portfolio_Value': portfolio_values_cwmr
})
csv_save_path_cwmr = os.path.join(data_folder, "CWMR_Model.csv")
portfolio_df_cwmr.to_csv(csv_save_path_cwmr, index=False)
