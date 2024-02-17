from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import CovarianceShrinkage
from pypfopt import expected_returns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from finrl import config_tickers
import os
import pandas as pd
import yfinance as yf
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl import config_tickers
from finrl.config import INDICATORS
import itertools
from stable_baselines3.common.logger import configure
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS, TRAINED_MODEL_DIR, RESULTS_DIR
from finrl.main import check_and_make_directories
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC, A2C, DDPG, PPO, TD3
from datetime import datetime

# random_seed = 72
# np.random.seed(random_seed)
# torch.manual_seed(random_seed)

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

# 数据收集
data = yf.download(successful_tickers, start="2008-12-20", end="2018-09-30")['Close']

# 初始化一个DataFrame来存储权重
weights_df = pd.DataFrame(index=data.index, columns=[ticker + "_weight" for ticker in successful_tickers])

# 定义历史时间窗口（比如2天）
window = 3

# 当前月份初始化
current_month = None

# 遍历每个交易日
for i in range(window, len(data)):
    current_date = data.index[i]

    # 检查是否是新的一个月
    if current_date.month != current_month:
        current_month = current_date.month  # 更新当前月份

        # 使用窗口数据
        window_data = data[i - window:i].dropna(axis=1)

        # 计算预期收益率和协方差
        mu = expected_returns.mean_historical_return(window_data)
        # S = risk_models.sample_cov(window_data)
        # 使用CovarianceShrinkage来获得协方差矩阵
        S = CovarianceShrinkage(window_data).ledoit_wolf()

        # 使用马科维茨模型
        ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))

        try:
            weights = ef.max_sharpe(risk_free_rate=0.02)
            cleaned_weights = ef.clean_weights()
        except ValueError as e:
            print(f"无法找到最优夏普比率组合: {e}")
            weights = ef.min_volatility()
            cleaned_weights = ef.clean_weights()

        # 存储权重
        for ticker in successful_tickers:
            if ticker in cleaned_weights:
                weight = cleaned_weights.get(ticker, 0)
                weights_df.loc[data.index[i], ticker + "_weight"] = np.maximum(0, weight)
            else:
                weights_df.loc[data.index[i], ticker + "_weight"] = 0
    else:
        # 如果不是新的一个月，保持上个月的权重
        weights_df.loc[data.index[i]] = weights_df.loc[data.index[i - 1]]

    # 打印夏普比率，即使未调整权重
    try:
        portfolio_performance = ef.portfolio_performance(risk_free_rate=0.02)
        sharpe_ratio = portfolio_performance[2]
    except ValueError as e:
        sharpe_ratio = "N/A"

    print(f"Date: {current_date.date()}, Sharpe Ratio: {sharpe_ratio}")

# 将权重数据合并到原始数据中
data_combined = pd.concat([data, weights_df], axis=1).fillna(0)
# 将数据集存入csv文件
data_combined.to_csv(f"{data_folder}/all_close_data.csv")

weights_df.dropna(inplace=True)
weights_df.to_csv(f"{data_folder}/weights_df.csv")

# data_combined 包含历史价格数据和对应的马科维茨模型计算出的权重
# tickers 是股票代码列表
# 生成X和y的列名
X_columns = successful_tickers  # 原始股票价格
y_columns = [ticker + '_weight' for ticker in successful_tickers]  # 对应的权重

# 提取X和y
X = data_combined[X_columns]
y = data_combined[y_columns]

# 根据指定的日期划分数据集
train = data_combined['2009-01-01':'2015-01-01']
test = data_combined['2015-01-01':'2016-01-01']
trade = data['2016-01-01':]

# 分别为训练集和测试集提取特征和标签
X_train = train[X_columns]
y_train = train[y_columns]
X_test = test[X_columns]
y_test = test[y_columns]

# 计算训练集的均值和标准差
X_train_mean = X_train.mean(axis=0)
X_train_std = X_train.std(axis=0)

# 标准化训练集和测试集
X_train_scaled = (X_train - X_train_mean) / X_train_std
X_test_scaled = (X_test - X_train_mean) / X_train_std

# 转换为 PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled.values.astype(np.float32))
y_train_tensor = torch.tensor(y_train.values.astype(np.float32))
X_test_tensor = torch.tensor(X_test_scaled.values.astype(np.float32))
y_test_tensor = torch.tensor(y_test.values.astype(np.float32))

# 创建 PyTorch 数据加载器
train_data = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)

# 预处理trade数据
X_trade = trade[X_columns]
X_trade_scaled = (X_trade - X_train_mean) / X_train_std
X_trade_tensor = torch.tensor(X_trade_scaled.values.astype(np.float32))

# 初始资金
initial_capital = 10000

# 计算每日股票收益率
stock_returns = trade.pct_change()

# 初始化资金曲线
capital = initial_capital
capital_curve = []
dates_for_plot = []

# 从2016年1月开始绘制资金曲线
start_date_for_plot = pd.Timestamp('2016-01-01')

# 遍历交易天数
for date in weights_df.index:
    if date >= start_date_for_plot and date in stock_returns.index:
        # 获取该日预测权重
        weights = weights_df.loc[date].values

        # 计算该日投资组合收益率
        daily_return = np.sum(stock_returns.loc[date] * weights)

        # 更新资金总额
        capital *= (1 + daily_return)

        # 记录日期和资金总额
        dates_for_plot.append(date)
        capital_curve.append(capital)

# 绘制资金曲线
plt.figure(figsize=(12, 6))
plt.plot(dates_for_plot, capital_curve, label='Capital Over Time')
plt.title('Markowitz Model Capital Curve (From 2016-01-01)')
plt.xlabel('Date')
plt.ylabel('Capital in USD')
plt.legend()
plt.savefig(f"{image_folder}/capital_curve.png")
plt.close()

# 创建一个包含日期和资本值的DataFrame
capital_curve_df = pd.DataFrame({
    'Date': dates_for_plot,
    'Capital': capital_curve
})

# 设置日期为索引
capital_curve_df.set_index('Date', inplace=True)

# 保存为CSV文件
csv_file_path = f"{data_folder}/Markowitz_knowledge_drill_capital_curve.csv"
capital_curve_df.to_csv(csv_file_path)

print(f"资金曲线数据已保存到 {csv_file_path}")
