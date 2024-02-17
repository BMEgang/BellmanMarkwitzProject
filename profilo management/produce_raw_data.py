import pandas as pd
import yfinance as yf
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import CovarianceShrinkage
from pypfopt import expected_returns
from finrl import config_tickers
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import pandas as pd
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
# from finrl import config_tickers
from finrl.config import INDICATORS
import itertools
from stable_baselines3.common.logger import configure
# from finrl.agents.stablebaselines3.models import DRLAgent
# from finrl.config import INDICATORS, TRAINED_MODEL_DIR, RESULTS_DIR
# from finrl.main import check_and_make_directories
# from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


tickers = config_tickers.DOW_30_TICKER
# tickers.append('^FVX')

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
# data['^FVX'] *= 10
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

weights_df.dropna(inplace=True)


TRAIN_START_DATE = '2009-01-01'
TRAIN_END_DATE = '2015-01-01'
TRADE_START_DATE = '2016-01-01'
TRADE_END_DATE = '2018-09-01'

df_raw = YahooDownloader(start_date=TRAIN_START_DATE,
                         end_date=TRADE_END_DATE,
                         ticker_list=config_tickers.DOW_30_TICKER).fetch_data()

fe = FeatureEngineer(use_technical_indicator=True,
                     tech_indicator_list=INDICATORS,
                     use_vix=True,
                     use_turbulence=True,
                     user_defined_feature=False)

processed = fe.preprocess_data(df_raw)

list_ticker = processed["tic"].unique().tolist()
list_date = list(pd.date_range(processed['date'].min(), processed['date'].max()).astype(str))
combination = list(itertools.product(list_date, list_ticker))

processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(processed, on=["date", "tic"],
                                                                          how="left")
processed_full = processed_full[processed_full['date'].isin(processed['date'])]
processed_full = processed_full.sort_values(['date', 'tic'])

processed_full = processed_full.fillna(0)

processed_full['weight'] = np.nan

processed_full.set_index('date', inplace=True)

# 截取weights_df以匹配processed_full的日期范围
start_date = processed_full.index.min()
end_date = processed_full.index.max()
weights_df_trimmed = weights_df.loc[start_date:end_date]

tics = processed_full['tic'].unique()
dates = processed_full.index.unique()

for i, tic in enumerate(tics):
    print(f"{i + 1}th, now processing {tic}, the date number is {len(dates)}")
    for date in dates:
        ticker_weight = tic + "_weight"
        if date in weights_df_trimmed.index and ticker_weight in weights_df_trimmed.columns:
            weight = weights_df_trimmed.loc[date, ticker_weight]

            if not pd.isnull(weight):
                processed_full.loc[
                    (processed_full.index == date) & (processed_full.tic == tic), ['weight']] = weight

processed_full.to_csv("/Users/ganghu/Desktop/pythonProject1/raw_data_new.csv")