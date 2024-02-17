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

# 以下的部分是用ddpg模型进行投资组合，最后把手里所拥有的资金画出图来
# 经过我的研究，ddpg的神经网络初始化在这个文件：/finrl_env/lib/python3.10/site-packages/stable_baselines3/td3/policies.py
# 我现在已经把它的初始网络给改了，

TRAIN_START_DATE = '2009-01-01'
TRAIN_END_DATE = '2015-01-01'
TRADE_START_DATE = '2016-01-01'
TRADE_END_DATE = '2018-09-01'

# 读取数据，
processed_full = pd.read_csv("/Users/ganghu/Desktop/pythonProject1/raw_data.csv", parse_dates=['date'])

# # 定义要保留的股票代码列表
tickers_to_keep = ['AAPL', 'CRM', 'V', 'UNH', 'BA']#
#
# # 筛选出符合条件的行
processed_full = processed_full[processed_full['tic'].isin(tickers_to_keep)]

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
print("begin to train")
trained_ddpg = agent.train_model(model=model_ddpg,
                             tb_log_name='ddpg',
                             total_timesteps=50000)#65000
trained_ddpg.save(model_folder + "/agent_ddpg")

# ************************************* backtesting *****************************
trained_ddpg = DDPG.load(model_folder + "/agent_ddpg")

e_trade_gym = StockTradingEnv(df = trade, risk_indicator_col='vix', **env_kwargs)

print("begin to predict")
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
df_actions_ddpg.to_csv(f"{data_folder}/actions_ddpg.csv")
# 绘制线图
plt.figure(figsize=(10, 6))
plt.plot(df['date'], df['account_value'])
plt.title('Account Value Over Time')
plt.xlabel('Date')
plt.ylabel('Account Value')
plt.grid(True)
plt.savefig(f'{image_folder}/final_image.png')  # 保存图表为 PNG 文件
