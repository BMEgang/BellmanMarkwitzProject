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


# ************************************* process data *****************************
check_and_make_directories([TRAINED_MODEL_DIR])

data_folder = "/Users/ganghu/Desktop/pythonProject1/data"

if not os.path.exists(data_folder):
    print(f"make {data_folder} success")
    os.makedirs(data_folder)

TRAIN_START_DATE = '2009-01-01'
TRAIN_END_DATE = '2015-01-01'
TRADE_START_DATE = '2016-01-01'
TRADE_END_DATE = '2018-09-01'
#
df_raw = YahooDownloader(start_date = TRAIN_START_DATE,
                     end_date = TRADE_END_DATE,
                     ticker_list = config_tickers.DOW_30_TICKER).fetch_data()
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
#
# processed_full.to_csv(f"{data_folder}/processed_full.csv")





processed_full = pd.read_csv("/Users/ganghu/Desktop/pythonProject1/data/after_processed_full.csv")

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
tmp_path = RESULTS_DIR + '/ddpg'
new_logger_ddpg = configure(tmp_path, ["stdout", "csv", "tensorboard"])
# Set new logger
model_ddpg.set_logger(new_logger_ddpg)
trained_ddpg = agent.train_model(model=model_ddpg,
                             tb_log_name='ddpg',
                             total_timesteps=50000)
trained_ddpg.save(TRAINED_MODEL_DIR + "/agent_ddpg")

# ************************************* backtesting *****************************

train = pd.read_csv(f'{data_folder}/train_data.csv')
trade = pd.read_csv(f'{data_folder}/trade_data.csv')

train = train.set_index(train.columns[0])
train.index.names = ['']
trade = trade.set_index(trade.columns[0])
trade.index.names = ['']

trained_ddpg = DDPG.load(TRAINED_MODEL_DIR + "/agent_ddpg")

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

df.to_csv(f"{data_folder}originate_ddpg.csv")
# 绘制线图
plt.figure(figsize=(10, 6))
plt.plot(df['date'], df['account_value'])
plt.title('Account Value Over Time')
plt.xlabel('Date')
plt.ylabel('Account Value')
plt.grid(True)
plt.savefig('account_value_over_time.png')  # 保存图表为 PNG 文件
