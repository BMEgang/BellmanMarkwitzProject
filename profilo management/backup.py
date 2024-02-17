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

# 第一步，你需要先把这个包给安上 !pip install git+https://github.com/AI4Finance-Foundation/FinRL.git

# 获取当前时间并格式化为字符串（例如 '2023-11-04_12-30-00'）
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# ***************************************** 这部分是进行模型预训练的 **************
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
#
# # 股票代码
# tickers = config_tickers.DOW_30_TICKER
# # 尝试下载数据，如果失败则从列表中移除对应的股票代码
# successful_tickers = []
# for ticker in tickers:
#     try:
#         data = yf.download(ticker, start="2008-12-20", end="2018-08-30")
#         if not data.empty:
#             successful_tickers.append(ticker)
#     except Exception as e:
#         print(f"Failed to download {ticker}: {e}")
#
# # 数据收集
# data = yf.download(successful_tickers, start="2008-12-20", end="2018-09-30")['Close']
#
# # 初始化一个DataFrame来存储权重
# weights_df = pd.DataFrame(index=data.index, columns=[ticker + "_weight" for ticker in successful_tickers])

# # 定义历史时间窗口（比如2天）
# window = 3
#
# # 当前月份初始化
# current_month = None
#
# # 遍历每个交易日
# for i in range(window, len(data)):
#     current_date = data.index[i]
#
#     # 检查是否是新的一个月
#     if current_date.month != current_month:
#         current_month = current_date.month  # 更新当前月份
#
#         # 使用窗口数据
#         window_data = data[i - window:i].dropna(axis=1)
#
#         # 计算预期收益率和协方差
#         mu = expected_returns.mean_historical_return(window_data)
#         # S = risk_models.sample_cov(window_data)
#         # 使用CovarianceShrinkage来获得协方差矩阵
#         S = CovarianceShrinkage(window_data).ledoit_wolf()
#
#         # 使用马科维茨模型
#         ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
#
#         try:
#             weights = ef.max_sharpe(risk_free_rate=0.02)
#             cleaned_weights = ef.clean_weights()
#         except ValueError as e:
#             print(f"无法找到最优夏普比率组合: {e}")
#             weights = ef.min_volatility()
#             cleaned_weights = ef.clean_weights()
#
#         # 存储权重
#         for ticker in successful_tickers:
#             if ticker in cleaned_weights:
#                 weight = cleaned_weights.get(ticker, 0)
#                 weights_df.loc[data.index[i], ticker + "_weight"] = np.maximum(0, weight)
#             else:
#                 weights_df.loc[data.index[i], ticker + "_weight"] = 0
#     else:
#         # 如果不是新的一个月，保持上个月的权重
#         weights_df.loc[data.index[i]] = weights_df.loc[data.index[i - 1]]
#
#     # 打印夏普比率，即使未调整权重
#     try:
#         portfolio_performance = ef.portfolio_performance(risk_free_rate=0.02)
#         sharpe_ratio = portfolio_performance[2]
#     except ValueError as e:
#         sharpe_ratio = "N/A"
#
#     print(f"Date: {current_date.date()}, Sharpe Ratio: {sharpe_ratio}")
#
# # 将权重数据合并到原始数据中
# data_combined = pd.concat([data, weights_df], axis=1).fillna(0)
# # 将数据集存入csv文件
# data_combined.to_csv(f"{data_folder}/all_close_data.csv")
#
# weights_df.dropna(inplace=True)
# weights_df.to_csv(f"{data_folder}/weights_df.csv")
#
# # data_combined 包含历史价格数据和对应的马科维茨模型计算出的权重
# # tickers 是股票代码列表
# # 生成X和y的列名
# X_columns = successful_tickers  # 原始股票价格
# y_columns = [ticker + '_weight' for ticker in successful_tickers]  # 对应的权重
#
# # 提取X和y
# X = data_combined[X_columns]
# y = data_combined[y_columns]
#
# # 根据指定的日期划分数据集
# train = data_combined['2009-01-01':'2015-01-01']
# test = data_combined['2015-01-01':'2016-01-01']
# trade = data['2016-01-01':]
#
# # 分别为训练集和测试集提取特征和标签
# X_train = train[X_columns]
# y_train = train[y_columns]
# X_test = test[X_columns]
# y_test = test[y_columns]
#
# # 计算训练集的均值和标准差
# X_train_mean = X_train.mean(axis=0)
# X_train_std = X_train.std(axis=0)
#
# # 标准化训练集和测试集
# X_train_scaled = (X_train - X_train_mean) / X_train_std
# X_test_scaled = (X_test - X_train_mean) / X_train_std
#
# # 转换为 PyTorch tensors
# X_train_tensor = torch.tensor(X_train_scaled.values.astype(np.float32))
# y_train_tensor = torch.tensor(y_train.values.astype(np.float32))
# X_test_tensor = torch.tensor(X_test_scaled.values.astype(np.float32))
# y_test_tensor = torch.tensor(y_test.values.astype(np.float32))
#
# # 创建 PyTorch 数据加载器
# train_data = TensorDataset(X_train_tensor, y_train_tensor)
# train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
#
# class Net(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(input_size, 128)  # 第一层
#         self.fc2 = nn.Linear(128, 64)  # 第二层
#         self.fc3 = nn.Linear(64, output_size)  # 输出层
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)  # 没有激活函数，直接输出
#         return x
#
#
# # 实例化模型
# model = Net(X_train_tensor.shape[1], y_train_tensor.shape[1])
#
# # 定义损失函数和优化器
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # 训练模型
# epochs = 200
# for epoch in range(epochs):
#     total_loss = 0
#
#     for inputs, targets in train_loader:
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#
#     average_loss = total_loss / len(train_loader)
#     print(f'Epoch [{epoch + 1}/{epochs}], Loss: {average_loss:.4f}')
#
#
# for param in model.parameters():
#     print(param)
#
# # 评估模型
# model.eval()
# with torch.no_grad():
#     raw_predictions = model(X_test_tensor)
# # 应用非负约束
# non_negative_predictions = torch.relu(raw_predictions)
#
# # 应用权重总和约束
# summed_predictions = non_negative_predictions.sum(dim=1, keepdim=True)
# normalized_predictions = non_negative_predictions / summed_predictions
#
# # 计算测试损失（注意：这里的损失计算可能需要根据新的输出进行调整）
# test_loss = criterion(normalized_predictions, y_test_tensor)
# print(f'Test Loss: {test_loss.item()}')
#
# # 计算RMSE
# rmse = np.sqrt(test_loss.item())
# print(f'Test RMSE: {rmse}')
#
# # 可视化预测与实际值
# y_pred_np = normalized_predictions.numpy()
# y_test_np = y_test_tensor.numpy()
#
# for i in range(y_pred_np.shape[1]):
#     plt.figure(figsize=(12, 6))
#     plt.plot(y_test_np[:, i], label='Actual')
#     plt.plot(y_pred_np[:, i], label='Predicted')
#     plt.title(f'Prediction vs Actual for {y.columns[i]}')
#     plt.xlabel('Time')
#     plt.ylabel('Weight')
#     plt.legend()
#     plt.savefig(f"{image_folder}/plot_{i}.png")  # 保存图像
#     plt.close()  # 关闭图像以释放内存
#
# model_save_path = f"{model_folder}/knowledgement_drill_model.pth"
# torch.save(model.state_dict(), model_save_path)
#
# # 预处理trade数据
# X_trade = trade[X_columns]
# X_trade_scaled = (X_trade - X_train_mean) / X_train_std
# X_trade_tensor = torch.tensor(X_trade_scaled.values.astype(np.float32))
#
# # 使用模型进行预测
# model.eval()
# with torch.no_grad():
#     trade_predictions = model(X_trade_tensor)
#     # 应用非负约束
#     trade_predictions_nn = torch.relu(trade_predictions)
#     # 应用权重总和约束
#     trade_summed_predictions = trade_predictions_nn.sum(dim=1, keepdim=True)
#     trade_normalized_predictions = trade_predictions_nn / trade_summed_predictions

# **************************************这个模块结束*****************************************************
# 上面的代码展示了对于神经网络的监督学习，利用马科维茨理论（你可以看成一个黑箱）让它学习投资组合，并且最后可以用来预测投资组合权重。
# 这个神经网络的输入是yfinance的收盘价（close），输出是投资比例。这个部分就是我想对于ddpg神经网络初始化训练的部分。


# 以下的部分是用ddpg模型进行投资组合，最后把手里所拥有的资金画出图来
# 经过我的研究，ddpg的神经网络初始化在这个文件：/finrl_env/lib/python3.10/site-packages/stable_baselines3/td3/policies.py 这个地址也可能不对，
# 你要看一下，现在你要做的事就是把上面那部分的预处理神经网络加到ddpg的agent初始化里面，这个坑的地方在于神经网络的输入输出维度，我搞不清楚。


TRAIN_START_DATE = '2009-01-01'
TRAIN_END_DATE = '2015-01-01'
TRADE_START_DATE = '2016-01-01'
TRADE_END_DATE = '2018-09-01'

# df_raw = YahooDownloader(start_date = TRAIN_START_DATE,
#                      end_date = TRADE_END_DATE,
#                      ticker_list = config_tickers.DOW_30_TICKER).fetch_data()
#
# fe = FeatureEngineer(use_technical_indicator=True,
#                      tech_indicator_list = INDICATORS,
#                      use_vix=True,
#                      use_turbulence=True,
#                      user_defined_feature = False)
#
# processed = fe.preprocess_data(df_raw)
#
# list_ticker = processed["tic"].unique().tolist()
# list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
# combination = list(itertools.product(list_date,list_ticker))
#
# processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
# processed_full = processed_full[processed_full['date'].isin(processed['date'])]
# processed_full = processed_full.sort_values(['date','tic'])
#
# processed_full = processed_full.fillna(0)

processed_full = pd.read_csv("/Users/ganghu/Desktop/pythonProject1/raw_data.csv", parse_dates=['date'])
# processed_full.set_index('date',inplace=True)

del processed_full['weight']

# 截取weights_df以匹配processed_full的日期范围
start_date = processed_full.index.min()
end_date = processed_full.index.max()

tics = processed_full['tic'].unique()
dates = processed_full.index.unique()

# Split the data for training and trading
train = data_split(processed_full, TRAIN_START_DATE,TRAIN_END_DATE)
trade = data_split(processed_full, TRADE_START_DATE,TRADE_END_DATE)
# Save data to csv file
train.to_csv(f'{data_folder}/train_data.csv')
trade.to_csv(f'{data_folder}/trade_data.csv')

# ************************************* build the environment *****************************
train = pd.read_csv(f'{data_folder}/train_data.csv')
train = train.set_index(train.columns[0])
train.index.names = ['']
stock_dimension = len(train.tic.unique())
state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

env_kwargs = {
    "hmax": 100,
    "initial_amount": 10000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4
}

e_train_gym = StockTradingEnv(df = train, **env_kwargs)
env_train, _ = e_train_gym.get_sb_env()

# ************************************* train RL agent *****************************
agent = DRLAgent(env = env_train)
model_ddpg = agent.get_model("ddpg")
# set up logger
tmp_path = result_folder
new_logger_ddpg = configure(tmp_path, ["stdout", "csv", "tensorboard"])
# Set new logger
model_ddpg.set_logger(new_logger_ddpg)
trained_ddpg = agent.train_model(model=model_ddpg,
                             tb_log_name='ddpg',
                             total_timesteps=60000)#50000
trained_ddpg.save(model_folder + "/agent_ddpg")

# ************************************* backtesting *****************************
trained_ddpg = DDPG.load(model_folder + "/agent_ddpg")

e_trade_gym = StockTradingEnv(df = trade, risk_indicator_col='vix', **env_kwargs)

df_account_value_ddpg, df_actions_ddpg = DRLAgent.DRL_prediction(
    model=trained_ddpg,
    environment = e_trade_gym)

print("training finished !")
# ************************************* plot result *****************************
print("try to plot")
df_result_ddpg = (
    df_account_value_ddpg.set_index(df_account_value_ddpg.columns[0])
)
print(f"df_account_value_ddpg head is {df_account_value_ddpg}")

# 假设 df 是你的数据帧
df = df_account_value_ddpg

# 转换日期列为 datetime 类型，以便更好地处理日期
df['date'] = pd.to_datetime(df['date'])

df.to_csv(f"{data_folder}/originate_ddpg.csv")
# 绘制线图
plt.figure(figsize=(10, 6))
plt.plot(df['date'], df['account_value'])
plt.title('Account Value Over Time')
plt.xlabel('Date')
plt.ylabel('Account Value')
plt.grid(True)
plt.savefig(f'{image_folder}/final_image.png')  # 保存图表为 PNG 文件
