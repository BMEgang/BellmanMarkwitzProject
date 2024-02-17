import yfinance as yf
import pandas as pd

# 股票代码列表
tickers = ['AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'GS',
           'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK',
           'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT']

# 设定时间范围
start_date = "2009-01-01"
end_date = "2018-01-01"

# 下载数据
data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# 计算每只股票在指定时间段内的总收益
total_returns = data.iloc[-1] / data.iloc[0] - 1

# 对收益进行排序并选择前五名
top_five_stocks = total_returns.nlargest(5)
print(top_five_stocks)
