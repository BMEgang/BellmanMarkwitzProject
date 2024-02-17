import pandas as pd


# 假设数据已经按照上述格式加载到DataFrame中
data = {
    'date': [
        '2017-11-01', '2017-11-02', '2017-11-03', '2017-11-06', '2017-11-07',
        '2017-11-08', '2017-11-09', '2017-11-10', '2017-11-13', '2017-11-14',
        '2017-11-15', '2017-11-16', '2017-11-17', '2017-11-20', '2017-11-21',
        '2017-11-22', '2017-11-24', '2017-11-27', '2017-11-28', '2017-11-29',
        '2017-11-30', '2017-12-01'
    ],
    'account_value': [
        14464.39761164090, 14429.059528373700, 14461.471221946700, 14429.870679878200, 14429.960330986000,
        14568.546812080400, 14452.418211006200, 14381.09328462980, 14394.962017082200, 14378.974300407400,
        14351.323572181700, 14466.208501838700, 14394.193908714300, 14433.811401390100, 14479.846195243800,
        14375.928354286200, 14439.811960243200, 14547.731678031900, 14776.265754722600, 14739.43611719510,
        14684.961044334400, 14762.99698259740
    ]
    # 'account_value': [
    #     18462.78, 18600.79, 18756.38, 18690.45, 18741.44,
    #     18571.78, 18641.23, 18598.15, 18737.93, 18654.41,
    #     18491.79, 18603.42, 18495.30, 18526.07, 18732.65,
    #     18611.34, 18724.73, 18715.06, 19043.83, 19636.31,
    #     20169.06, 20046.45
    # ]
}

df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# 计算每日收益率
df['daily_returns'] = df['account_value'].pct_change()

# 计算波动性（标准差）
volatility = df['daily_returns'].std()

print(volatility)