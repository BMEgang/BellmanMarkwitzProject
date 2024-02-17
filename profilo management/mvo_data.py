import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pypfopt.efficient_frontier import EfficientFrontier

draw_folder = "/Users/ganghu/Desktop/pythonProject1/draw"

def process_df_for_mvo(df):
    return df.pivot(index="date", columns="tic", values="close")


def StockReturnsComputing(StockPrice, Rows, Columns):
    StockReturn = np.zeros([Rows - 1, Columns])
    for j in range(Columns):  # j: Assets
        for i in range(Rows - 1):  # i: Daily Prices
            StockReturn[i, j] = ((StockPrice[i + 1, j] - StockPrice[i, j]) / StockPrice[i, j]) * 100

    return StockReturn

train = pd.read_csv(f'{draw_folder}/train_data.csv')
trade = pd.read_csv(f'{draw_folder}/trade_data.csv')

train = train.set_index(train.columns[0])
train.index.names = ['']
trade = trade.set_index(trade.columns[0])
trade.index.names = ['']

# ************************************* plot MVO result *****************************
StockData = process_df_for_mvo(train)
TradeData = process_df_for_mvo(trade)

# compute asset returns
arStockPrices = np.asarray(StockData)
[Rows, Cols] = arStockPrices.shape
arReturns = StockReturnsComputing(arStockPrices, Rows, Cols)

# compute mean returns and variance covariance matrix of returns
meanReturns = np.mean(arReturns, axis=0)
covReturns = np.cov(arReturns, rowvar=False)

# set precision for printing results
np.set_printoptions(precision=3, suppress=True)

ef_mean = EfficientFrontier(meanReturns, covReturns, weight_bounds=(0, 0.1))
raw_weights_mean = ef_mean.max_sharpe()
cleaned_weights_mean = ef_mean.clean_weights()
mvo_weights = np.array([10000 * cleaned_weights_mean[i] for i in range(len(cleaned_weights_mean))])

LastPrice = np.array([1/p for p in StockData.tail(1).to_numpy()[0]])
Initial_Portfolio = np.multiply(mvo_weights, LastPrice)

Portfolio_Assets = TradeData @ Initial_Portfolio
MVO_result = pd.DataFrame(Portfolio_Assets, columns=["Mean Var"])

print("try to plot")
df_result_mvo = (
    MVO_result.set_index(MVO_result.columns[0])
)
print(f"df_account_value_ddpg head is {MVO_result}")

# 假设 df 是你的数据帧
df = MVO_result

df.to_csv(f"{draw_folder}/MVO.csv")