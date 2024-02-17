import pandas as pd

# 读取CSV文件
data = pd.read_csv("/Users/ganghu/Desktop/pythonProject1/draw/richard4.csv")
# 将日期列转换为datetime类型
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')

# 设置日期为索引
data.set_index('date', inplace=True)

# 计算每月的第一天和最后一天的值
monthly_first = data.resample('MS').first()['account_value']
monthly_last = data.resample('M').last()['account_value']

monthly_first = monthly_first.reset_index()
monthly_last = monthly_last.reset_index()

# 计算每个月的值变化
monthly_change = monthly_last['account_value'] - monthly_first['account_value']

# 筛选出上涨的月份，并找出上涨最大的月份
# max_increase = monthly_change[monthly_change > 0].max()
# max_increase_month = monthly_change.idxmax()
#
print("cao")
# print(f"最大上涨值: {max_increase}")
# print(f"最大上涨的月份: {max_increase_month.strftime('%Y-%m')}")
