import numpy as np
import pandas as pd

# 载入数据
file_weights_df = "/Users/ganghu/Desktop/pythonProject1/data/weights_df.csv"
file_processed_full = "/Users/ganghu/Desktop/pythonProject1/data/processed_full.csv"

weights_df = pd.read_csv(file_weights_df)
processed_full = pd.read_csv(file_processed_full, index_col='date')

weights_df.set_index('Date', inplace=True)

# 截取weights_df以匹配processed_full的日期范围
start_date = processed_full.index.min()
end_date = processed_full.index.max()
weights_df_trimmed = weights_df.loc[start_date:end_date]

tics = processed_full['tic'].unique()
dates = processed_full.index.unique()

for i,tic in enumerate(tics):
    print(f"{i+1}th, now processing {tic}, the date number is {len(dates)}")
    for date in dates:
        # processed_full[(processed_full.index == date) & (processed_full.tic == tic)]
        # weights_df_trimmed[(weights_df_trimmed.index == date)]['AAPL_weight'].iloc[0]

        ticker_weight = tic+"_weight"
        # processed_full[(processed_full.index == date) & (processed_full.tic == tic)][['open', 'high', 'low', 'close', 'volume', 'day', 'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma', 'vix', 'turbulence']] *= (
        #     weights_df_trimmed[(weights_df_trimmed.index == date)][ticker_weight].iloc)[0]
        if date in weights_df_trimmed.index and ticker_weight in weights_df_trimmed.columns:
            weight = weights_df_trimmed.loc[date, ticker_weight]

            if not pd.isnull(weight):
                processed_full.loc[(processed_full.index == date) & (processed_full.tic == tic), ['open', 'high', 'low', 'close', 'volume']] *= (1 + weight)
# 'open', 'high', 'low', 'close', 'volume', 'day', 'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma', 'vix', 'turbulence'
# 将0替换为NaN
processed_full.replace(0, 1, inplace=True)

# 删除所有值为NaN的行
processed_full.dropna(inplace=True)

processed_full.to_csv("/Users/ganghu/Desktop/pythonProject1/data/after_processed_full.csv")
print("hello")


